---
name: run
description: Start the HeadMark dev server (python3 app.py on port 8002 with hot-reload) and confirm it's accepting requests.
---

1. Check if port 8002 is already in use: `lsof -ti:8002`. If a process is running, report it to the user and ask whether to kill it first.
2. Activate the venv: `source venv/bin/activate` (from the project root).
3. Start the server in the background: `python3 app.py &` — capture the PID.
4. Wait up to 5 seconds for the server to be ready by polling `curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/` until it returns `200`.
5. Report the URL (`http://localhost:8002`) and the server PID to the user so they can stop it later with `kill <PID>`.
