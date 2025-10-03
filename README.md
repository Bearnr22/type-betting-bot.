# Betting Assistant Bot (Telegram) — Phase A

**Created:** 2025-10-03 11:42 UTC

A realistic, legal Telegram bot that:
- Models **football** matches (Poisson) and gives 1X2, O2.5, BTTS probabilities.
- Ranks **horse races** from uploaded CSV cards with transparent scoring → % win.
- Schedules a daily drop at your local time (Europe/Dublin).

## Commands

- `/start` — link your chat and subscribe for the daily drop.
- `/settime HH:MM` — schedule the daily drop (e.g., `08:30`). Reschedules immediately.
- `/football_addleague <CODE>` — track a league (e.g., `PL`, `PD`, `SA`, `BL1`, `FL1`, `CL`).
- `/football_today` — today's tracked matches with model probabilities.
- `/horse_import` — upload a CSV racecard (see template below).
- `/horse_rank` — ranks each race with top 3 picks and % win.
- `/export_template` — sends a sample CSV template for horse cards.

### Horse CSV columns (minimum)
```
race_id,race_time,course,going,dist_furlongs,horse,draw,runners,or,weight_lbs,age,last3,trainer_win%,jockey_win%,days_since_run,cls,cls_change,going_pref,pace_pref
```

## Quick Start (Local)

1) Install Python 3.11+
2) `pip install -r requirements.txt`
3) Copy `.env.example` to `.env` and fill tokens.
4) `python app.py`
5) In Telegram: find your bot → `/start`, then e.g. `/football_addleague PL`, `/football_today`

## Docker (Railway/Render/VPS)
```
docker build -t betting-bot .
docker run -e TELEGRAM_BOT_TOKEN=... -e FOOTBALL_DATA_API_KEY=... -e TZ=Europe/Dublin betting-bot
```

## Notes & Limits
- Football data via football-data.org free tier (limited comps & rate limits).
- Horse model needs your CSVs (public/proprietary data not bundled).
- This is **not** guaranteed to beat markets. Use small, fixed staking.
