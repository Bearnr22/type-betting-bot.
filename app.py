
import os, io, math
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
import requests
from dateutil import tz
from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ConversationHandler,
    ContextTypes, filters
)

# ---------- ENV & GLOBAL STORAGE ----------
TZ = os.getenv("TZ", "Europe/Dublin")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
FD_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
assert BOT_TOKEN, "Set TELEGRAM_BOT_TOKEN"
assert FD_KEY, "Set FOOTBALL_DATA_API_KEY"

STATE_IMPORT_HORSE = 10

storage = {
    "subscribers": set(),           # chat_ids
    "leagues": set(["PL"]),         # football-data.org codes
    "horse_cards": {},              # race_id -> DataFrame
    "drop_time": "08:30"
}

# ---------- UTIL ----------
def now_local():
    return datetime.now(tz.gettz(TZ))

def parse_hhmm(hhmm: str):
    try:
        h, m = hhmm.split(":")
        return time(hour=int(h), minute=int(m))
    except Exception:
        return None

def reply_chunks(text, chunk=3500):
    return [text[i:i+chunk] for i in range(0, len(text), chunk)]

# ---------- FOOTBALL (football-data.org) ----------
FD_ENDPOINT = "https://api.football-data.org/v4"

def fd_get(path, params=None):
    headers = {"X-Auth-Token": FD_KEY}
    r = requests.get(FD_ENDPOINT+path, headers=headers, params=params or {}, timeout=15)
    r.raise_for_status()
    return r.json()

def football_today_matches():
    # football-data filters by date strings (YYYY-MM-DD)
    d = now_local().date().isoformat()
    matches = []
    for code in storage["leagues"]:
        try:
            data = fd_get(f"/competitions/{code}/matches", {"dateFrom": d, "dateTo": d})
            for m in data.get("matches", []):
                if m.get("status") in ("TIMED","SCHEDULED"):
                    matches.append(m)
                elif m.get("status") == "FINISHED":
                    matches.append(m)  # include finished for completeness
        except Exception:
            continue
    return matches

def team_recent_stats(team_id, last_n=10):
    # last 50 finished, take head(last_n)
    try:
        data = fd_get(f"/teams/{team_id}/matches", {"status":"FINISHED", "limit": 50})
    except Exception:
        return {"att":1.1,"def":1.1,"home_boost":0.15}
    rows = []
    for m in data.get("matches", []):
        th = m["homeTeam"]["id"]; ta = m["awayTeam"]["id"]
        sc = m.get("score",{}).get("fullTime",{})
        hg = sc.get("home"); ag = sc.get("away")
        if hg is None or ag is None: continue
        home = (th == team_id)
        gf = hg if home else ag
        ga = ag if home else hg
        rows.append({"home": home, "gf": gf, "ga": ga})
    df = pd.DataFrame(rows).head(last_n)
    if df.empty:
        return {"att":1.1,"def":1.1,"home_boost":0.15}
    att = max(df["gf"].mean(), 0.2)
    deff = max(df["ga"].mean(), 0.2)
    return {"att":att, "def":deff, "home_boost":0.15}

def poisson_matrix(lh, la, max_goals=10):
    # Poisson PMF via numpy for stability
    import math
    probs = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        # P(Home=i)
        ph = math.exp(-lh) * (lh**i) / math.factorial(i)
        for j in range(max_goals+1):
            pa = math.exp(-la) * (la**j) / math.factorial(j)
            probs[i,j] = ph*pa
    probs /= probs.sum()
    return probs

def model_match(m):
    hid = m["homeTeam"]["id"]; aid = m["awayTeam"]["id"]
    hs = team_recent_stats(hid); as_ = team_recent_stats(aid)
    league_avg = 2.6
    base = league_avg/2
    lambda_home = base * (hs["att"]/as_["def"]) * (1.0 + hs["home_boost"])
    lambda_away = base * (as_["att"]/hs["def"])

    mat = poisson_matrix(lambda_home, lambda_away, 10)
    home = sum(mat[i, j] for i in range(11) for j in range(11) if i>j)
    draw = sum(mat[i, i] for i in range(11))
    away = sum(mat[i, j] for i in range(11) for j in range(11) if i<j)
    o25 = sum(mat[i,j] for i in range(11) for j in range(11) if i+j>=3)
    btts = sum(mat[i,j] for i in range(1,11) for j in range(1,11))
    return {
        "home_xg": lambda_home, "away_xg": lambda_away,
        "p_home": float(home), "p_draw": float(draw), "p_away": float(away),
        "p_o25": float(o25), "p_btts": float(btts)
    }

# ---------- HORSE MODEL ----------
DRAW_BIAS = {
    ("Chester","5f"): {"low": +0.08, "mid": 0.0, "high": -0.05},
    ("Thirsk","6f"): {"low": +0.03, "mid": 0.0, "high": -0.02},
}

def draw_bucket(draw, total):
    if total<=0: return "mid"
    tercile = total/3
    if draw<=math.ceil(tercile): return "low"
    if draw<=math.ceil(2*tercile): return "mid"
    return "high"

def parse_last3(s):
    s = str(s)
    pts=0
    for ch in s:
        if ch.isdigit():
            val = int(ch)
            pts += max(0, 5 - val)  # 1st=4, 2nd=3, 3rd=2, 4th=1
    return pts

def horse_score(row):
    score = 0.0
    score += parse_last3(row.get("last3","")) * 0.8
    cc = float(row.get("cls_change",0) or 0)  # -1 means class drop
    score += (-cc)*0.7
    orv = float(row.get("or",0) or 0)
    w = float(row.get("weight_lbs",0) or 0)
    score += (orv - w/3.0)*0.05
    going = str(row.get("going","")).lower()
    pref = str(row.get("going_pref","")).lower()
    if pref and going:
        score += 1.2 if pref in going else -0.6
    course = str(row.get("course",""))
    try:
        distf = f'{int(float(row.get("dist_furlongs",0) or 0))}f'
    except Exception:
        distf = "0f"
    try:
        draw = int(float(row.get("draw",0) or 0))
    except Exception:
        draw = 0
    try:
        runners = int(float(row.get("runners",0) or 0))
    except Exception:
        runners = 0
    bias = DRAW_BIAS.get((course, distf))
    if bias and draw>0 and runners>0:
        buck = draw_bucket(draw, runners)
        score += bias[buck]*10
    t = float(row.get("trainer_win%",0) or 0); j = float(row.get("jockey_win%",0) or 0)
    score += (t*0.5 + j*0.4)
    try:
        d = int(float(row.get("days_since_run",0) or 0))
    except Exception:
        d = 0
    if d>120: score -= 1.0
    pace = str(row.get("pace_pref","")).lower()
    if pace=="front" and draw>0 and runners>0 and draw<=max(1, runners//5): score += 0.7
    return score

def scores_to_probs(df):
    s = df["score"].astype(float)
    z = (s - s.mean())/(s.std() + 1e-6)
    exps = np.exp(z)
    probs = exps / exps.sum()
    return probs

# ---------- TELEGRAM HANDLERS ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    storage["subscribers"].add(chat_id)
    await update.message.reply_text(
        "Subscribed for daily drop. Use /settime HH:MM, /football_addleague PL, /football_today, /horse_import, /horse_rank, /export_template."
    )

async def cmd_settime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Usage: /settime HH:MM (24h)")
    t = parse_hhmm(context.args[0])
    if not t:
        return await update.message.reply_text("Invalid time. Example: /settime 08:30")
    storage["drop_time"] = context.args[0]
    # reschedule
    job_queue = context.application.job_queue
    # clear previous
    for job in job_queue.get_jobs_by_name("daily_drop"):
        job.schedule_removal()
    # schedule next daily
    job_queue.run_daily(callback=daily_drop, time=t, name="daily_drop", chat_id=None, data=None)
    await update.message.reply_text(f"Daily drop set to {storage['drop_time']} ({TZ}).")

async def cmd_football_addleague(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await update.message.reply_text("Usage: /football_addleague <CODE>. Example: /football_addleague PL")
    code = context.args[0].upper()
    storage["leagues"].add(code)
    await update.message.reply_text(f"Tracking league: {code}")

async def cmd_football_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ms = football_today_matches()
    if not ms:
        return await update.message.reply_text("No tracked matches today (or API limit).")
    out = []
    for m in ms:
        try:
            res = model_match(m)
        except Exception:
            continue
        line = (
            f"{m['competition']['name']}: {m['homeTeam']['name']} vs {m['awayTeam']['name']}\n"
            f"Home {res['p_home']:.2%} | Draw {res['p_draw']:.2%} | Away {res['p_away']:.2%}\n"
            f"O2.5 {res['p_o25']:.2%} | BTTS {res['p_btts']:.2%}\n"
        )
        out.append(line)
    text = "\n".join(out) if out else "No modelable matches."
    for chunk in reply_chunks(text):
        await update.message.reply_text(chunk)

async def cmd_export_template(update: Update, context: ContextTypes.DEFAULT_TYPE):
    csv = ("race_id,race_time,course,going,dist_furlongs,horse,draw,runners,or,weight_lbs,age,last3,trainer_win%,jockey_win%,days_since_run,cls,cls_change,going_pref,pace_pref\n"
           "123,14:30,Chester,Good,5,Speedster,2,10,92,133,4,321,12,10,21,2,-1,good,front\n"
           "123,14:30,Chester,Good,5,Grinder,6,10,88,131,5,245,14,9,14,2,0,good to soft,stalk\n")
    bio = io.BytesIO(csv.encode("utf-8"))
    bio.name = "horse_card_template.csv"
    await update.message.reply_document(document=bio, caption="Upload files like this via /horse_import.")

async def cmd_horse_import(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Upload racecard CSV now (one file per meeting).")
    return STATE_IMPORT_HORSE

async def horse_receive_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc or not doc.file_name.endswith(".csv"):
        await update.message.reply_text("Please send a .csv file.")
        return ConversationHandler.END
    f = await doc.get_file()
    by = await f.download_as_bytearray()
    df = pd.read_csv(io.BytesIO(by))
    if "race_id" not in df.columns or "horse" not in df.columns:
        await update.message.reply_text("CSV must include at least: race_id, horse.")
        return ConversationHandler.END
    for rid, grp in df.groupby("race_id"):
        storage["horse_cards"][str(rid)] = grp.reset_index(drop=True)
    await update.message.reply_text(f"Imported {len(storage['horse_cards'])} races.")
    return ConversationHandler.END

async def cmd_horse_rank(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not storage["horse_cards"]:
        return await update.message.reply_text("No races loaded. Use /horse_import to upload CSV.")
    out=[]
    for rid, df in storage["horse_cards"].items():
        tmp = df.copy()
        tmp["score"] = tmp.apply(horse_score, axis=1)
        tmp["p_win"] = scores_to_probs(tmp)
        tmp = tmp.sort_values("p_win", ascending=False).reset_index(drop=True)
        head = tmp.head(3)[["race_time","course","horse","p_win","score"]].fillna("")
        lines=[f"Race {rid} {str(head.iloc[0]['race_time'])} @ {head.iloc[0]['course']}"]
        for i,row in head.iterrows():
            lines.append(f"  {i+1}. {row['horse']} — {row['p_win']:.1%} (score {row['score']:.2f})")
        out.append("\n".join(lines))
    text="\n\n".join(out)
    for chunk in reply_chunks(text):
        await update.message.reply_text(chunk)

# ---------- DAILY DROP ----------
async def daily_drop(context: ContextTypes.DEFAULT_TYPE):
    # For each subscriber: send football and (if exists) horse ranking
    ms = football_today_matches()
    foot_lines = []
    for m in ms:
        try:
            res = model_match(m)
            line = (
                f"{m['competition']['name']}: {m['homeTeam']['name']} vs {m['awayTeam']['name']}\n"
                f"Home {res['p_home']:.0%} | Draw {res['p_draw']:.0%} | Away {res['p_away']:.0%} | BTTS {res['p_btts']:.0%} | O2.5 {res['p_o25']:.0%}"
            )
            foot_lines.append(line)
        except Exception:
            continue
    foot_text = "*Football (model %)*\n" + ("\n".join(foot_lines) if foot_lines else "No tracked matches.")
    horse_text = ""
    if storage["horse_cards"]:
        out=[]
        for rid, df in storage["horse_cards"].items():
            tmp = df.copy()
            tmp["score"] = tmp.apply(horse_score, axis=1)
            tmp["p_win"] = scores_to_probs(tmp)
            tmp = tmp.sort_values("p_win", ascending=False).reset_index(drop=True)
            head = tmp.head(3)[["race_time","course","horse","p_win"]]
            title = f"Race {rid} {str(head.iloc[0]['race_time'])} @ {head.iloc[0]['course']}"
            picks = " | ".join([f"{row['horse']} {row['p_win']:.0%}" for _,row in head.iterrows()])
            out.append(f"{title}\n{picks}")
        horse_text = "\n\n*Horse Racing (top 3 per race)*\n" + "\n\n".join(out)
    final = foot_text + ("\n\n" + horse_text if horse_text else "")
    for chat_id in list(storage["subscribers"]):
        try:
            for chunk in reply_chunks(final):
                await context.bot.send_message(chat_id=chat_id, text=chunk)
        except Exception:
            # if delivery fails (e.g., blocked bot), drop the subscriber
            storage["subscribers"].discard(chat_id)

def build_app():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("settime", cmd_settime))
    app.add_handler(CommandHandler("football_addleague", cmd_football_addleague))
    app.add_handler(CommandHandler("football_today", cmd_football_today))
    app.add_handler(CommandHandler("export_template", cmd_export_template))

    conv = ConversationHandler(
        entry_points=[CommandHandler("horse_import", cmd_horse_import)],
        states={STATE_IMPORT_HORSE: [MessageHandler(filters.Document.ALL, horse_receive_file)]},
        fallbacks=[],
    )
    app.add_handler(conv)
    app.add_handler(CommandHandler("horse_rank", cmd_horse_rank))

    # schedule initial daily drop at default time
    t = parse_hhmm(storage["drop_time"]) or time(8,30)
    app.job_queue.run_daily(callback=daily_drop, time=t, name="daily_drop")
    return app

if __name__ == "__main__":
    app = build_app()
    app.run_polling()
