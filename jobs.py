"""Bounded single-worker queue; session-scoped access and expiring files."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent

class Jobs:
    def __init__(self, ttl=3600, timeout=1200):
        base = os.environ.get('COREMAP_DATA')
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='coremap-', dir=base))
        self.ttl, self.timeout = ttl, timeout
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobs = {}
        threading.Thread(target=self._janitor, daemon=True).start()

    def _remove(self, folder):
        folder = Path(folder).resolve()
        if folder.parent != self.root.resolve():
            raise ValueError('Refusing to remove a directory outside job storage')
        shutil.rmtree(folder, ignore_errors=True)

    def _janitor(self):
        while True:
            time.sleep(60)
            self.cleanup()

    def cleanup(self):
        with self.lock:
            expired = [key for key, job in self.jobs.items() if job['state'] in ('complete', 'error') and time.time() - job['finished'] > self.ttl]
            for key in expired:
                self._remove(self.jobs[key]['folder'])
                del self.jobs[key]

    def submit(self, owner, mode, side, files):
        self.cleanup()
        if mode not in ('native', 'mni') or side not in ('Left', 'Right'):
            raise ValueError('Invalid mode or side')
        required = ('cbf', 'core') if mode == 'native' else ('core',)
        if any(files.get(k) is None for k in required):
            raise ValueError('Required image is missing')
        with self.lock:
            active = [j for j in self.jobs.values() if j['state'] in ('queued', 'running')]
            if any(j['owner'] == owner for j in active):
                raise ValueError('A task from this session is still active. / 本会话已有任务，请等待完成。')
            if len(active) >= 3:
                raise ValueError('Queue is full; retry later. / 队列已满，请稍后再试。')
            key = uuid.uuid4().hex
            folder = self.root / key
            folder.mkdir()
            try:
                for name in required:
                    upload = files[name]
                    upload.seek(0)
                    size = 0
                    with (folder/(name+'.upload')).open('wb') as output:
                        while chunk := upload.read(1024*1024):
                            size += len(chunk)
                            if size > 80*1024*1024:
                                raise ValueError('Each image must be at most 80 MB')
                            output.write(chunk)
                    upload.seek(0)
                    if size == 0:
                        raise ValueError('Empty file')
            except Exception:
                self._remove(folder)
                raise
            self.jobs[key] = dict(owner=owner, folder=folder, state='queued', mode=mode, side=side, created=time.time())
            self.pool.submit(self._run, key)
            return key

    def _run(self, key):
        with self.lock:
            job = self.jobs[key]
            job.update(state='running', started=time.time())
            print('CoreMap worker started',flush=True)
        folder = job['folder']
        outcome = {}
        try:
            with (folder/'worker.log').open('w', encoding='utf-8') as log:
                command = [sys.executable, '-u', str(ROOT/'worker.py'), str(folder), job['mode'], job['side']]
                done = subprocess.run(command, stdout=log, stderr=log, timeout=self.timeout,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                    env={**os.environ, 'PYTHONIOENCODING':'utf-8', 'MPLCONFIGDIR':str(self.root/'mpl')})
            if done.returncode:
                err = folder/'error.json'
                raise ValueError(json.loads(err.read_text('utf-8'))['error'] if err.exists() else 'Processing failed. Please verify the input images.')
            result = json.loads((folder/'result.json').read_text('utf-8'))
            result.pop('transforms', None)
            outcome = dict(state='complete', result=result)
        except subprocess.TimeoutExpired:
            outcome = dict(state='error', error='Processing exceeded 20 minutes. / 处理超过20分钟，任务已停止。')
        except Exception as error:
            outcome = dict(state='error', error=str(error))
        finally:
            # Keep only the derived downloads; source uploads and transform files are deleted.
            keep = {'Core_MNI.nii.gz', 'qc.png'} if outcome.get('state') == 'complete' else set()
            try:
                for path in folder.iterdir():
                    if path.is_file() and path.name not in keep:
                        path.unlink(missing_ok=True)
            except OSError:
                outcome = dict(state='error', error='Unable to clear temporary source files; please restart the service.')
            with self.lock:
                job.update(outcome)
                job['finished'] = time.time()
                print('CoreMap worker finished',outcome.get('state'), 'seconds',round(job['finished']-job['started']),flush=True)

    def get(self, key, owner):
        with self.lock:
            job = self.jobs.get(key)
            if not job or job['owner'] != owner:
                return None
            result = {k:v for k,v in job.items() if k != 'owner'}
            log = job['folder']/'worker.log'
            if log.exists():
                lines = log.read_text('utf-8',errors='replace').splitlines()
                for line in reversed(lines):
                    if line.startswith(('Registering CBF','Transforming Core','Building alignment')):
                        result['stage'] = line
                        break
            return result

    def forget(self, key, owner):
        with self.lock:
            job = self.get(key, owner)
            if job and job['state'] in ('complete','error'):
                self._remove(job['folder'])
                del self.jobs[key]
