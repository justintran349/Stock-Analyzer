#!/bin/bash
# Runs scripts/daily.py every weekday at 1:37 pm Pacific (after the 4 pm Eastern close),
# and next-hour picks every hour during market hours. macOS launchd.
# Note: background jobs can't read ~/Desktop without Full Disk Access. If data/daily.log says
# "Operation not permitted", move this folder to ~/Projects and re-run this script.
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$DIR/.venv/bin/python"
AGENTS="$HOME/Library/LaunchAgents"
mkdir -p "$AGENTS" "$DIR/data"

cat > "$AGENTS/com.stockanalyzer.daily.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.stockanalyzer.daily</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$DIR/scripts/daily.py</string></array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StartCalendarInterval</key><array>
    $(for d in 1 2 3 4 5; do echo "<dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>13</integer><key>Minute</key><integer>37</integer></dict>"; done)
  </array>
  <key>StandardOutPath</key><string>$DIR/data/daily.log</string>
  <key>StandardErrorPath</key><string>$DIR/data/daily.log</string>
</dict></plist>
EOF

cat > "$AGENTS/com.stockanalyzer.hourly.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.stockanalyzer.hourly</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$DIR/scripts/daily.py</string><string>--hourly</string></array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StartCalendarInterval</key><array>
    $(for d in 1 2 3 4 5; do for h in 7 8 9 10 11 12; do echo "<dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>$h</integer><key>Minute</key><integer>33</integer></dict>"; done; done)
  </array>
  <key>StandardOutPath</key><string>$DIR/data/hourly.log</string>
  <key>StandardErrorPath</key><string>$DIR/data/hourly.log</string>
</dict></plist>
EOF

for j in daily hourly; do
  launchctl unload "$AGENTS/com.stockanalyzer.$j.plist" 2>/dev/null || true
  launchctl load "$AGENTS/com.stockanalyzer.$j.plist"
done
echo "Installed. Daily run weekdays 1:37 pm, next-hour picks at :33 past 7am–12pm (your Mac's local time)."
echo "Logs: $DIR/data/daily.log and hourly.log. Remove with: launchctl unload ~/Library/LaunchAgents/com.stockanalyzer.*.plist"
