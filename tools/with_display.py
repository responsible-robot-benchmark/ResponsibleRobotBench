"""Run a command on a private Xvfb display; clean up only the server we start."""
import os
from pathlib import Path
import selectors
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
prefix=ROOT/'.runtime/xvfb'
env=dict(os.environ)
env['LD_LIBRARY_PATH']=str(prefix/'usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
server=subprocess.Popen([str(prefix/'usr/bin/Xvfb'),'-displayfd','1','-screen','0','1280x1024x24','-nolisten','tcp'],stdout=subprocess.PIPE,env=env,text=True)
try:
 selector=selectors.DefaultSelector();selector.register(server.stdout,selectors.EVENT_READ)
 if not selector.select(timeout=10):raise RuntimeError('Xvfb did not become ready within 10 seconds')
 display=server.stdout.readline().strip()
 if not display.isdigit():raise RuntimeError('Xvfb failed to allocate a display')
 env['DISPLAY']=':'+display;env['QT_QPA_PLATFORM']='xcb';env['LIBGL_ALWAYS_SOFTWARE']='1'
 result=subprocess.run(sys.argv[1:],env=env)
 raise SystemExit(result.returncode)
finally:
 server.terminate()
 try:server.wait(timeout=5)
 except subprocess.TimeoutExpired:server.kill();server.wait()
