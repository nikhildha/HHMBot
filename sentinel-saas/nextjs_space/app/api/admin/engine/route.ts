import { NextResponse } from 'next/server';
import { spawn, execSync } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

export const dynamic = 'force-dynamic';

// ─── Project root (HMMBOT/) — two levels up from nextjs_space ───────────────
const PROJECT_ROOT = path.resolve(process.cwd(), '..', '..');
const DATA_DIR = path.join(PROJECT_ROOT, 'data');
const LOG_FILE = path.join(DATA_DIR, 'bot.log');
const PID_FILE = path.join(DATA_DIR, '.engine.pid');

// ─── Helpers ────────────────────────────────────────────────────────────────

function isProcessRunning(pid: number): boolean {
    try {
        process.kill(pid, 0); // signal 0 = check existence
        return true;
    } catch {
        return false;
    }
}

function getStoredPid(): number | null {
    try {
        if (fs.existsSync(PID_FILE)) {
            const pid = parseInt(fs.readFileSync(PID_FILE, 'utf-8').trim(), 10);
            if (!isNaN(pid) && isProcessRunning(pid)) return pid;
            // Stale PID file — clean up
            fs.unlinkSync(PID_FILE);
        }
    } catch { /* ignore */ }
    return null;
}

function getLogTail(lines = 15): string[] {
    try {
        if (!fs.existsSync(LOG_FILE)) return [];
        const content = fs.readFileSync(LOG_FILE, 'utf-8');
        const allLines = content.split('\n').filter(Boolean);
        return allLines.slice(-lines);
    } catch {
        return [];
    }
}

function getEngineUptime(pid: number): string {
    try {
        // macOS: ps -o etime= gives elapsed time
        const raw = execSync(`ps -o etime= -p ${pid}`, { encoding: 'utf-8' }).trim();
        return raw || 'unknown';
    } catch {
        return 'unknown';
    }
}

// Detect python binary (venv or system)
function getPythonBin(): string {
    const venvPython = path.join(PROJECT_ROOT, '.venv', 'bin', 'python');
    if (fs.existsSync(venvPython)) return venvPython;
    // Try python3 then python
    try {
        execSync('which python3', { encoding: 'utf-8' });
        return 'python3';
    } catch {
        return 'python';
    }
}

// ─── GET: Engine status ─────────────────────────────────────────────────────

export async function GET() {
    const pid = getStoredPid();
    const running = pid !== null;

    return NextResponse.json({
        status: running ? 'running' : 'stopped',
        pid: pid,
        uptime: running ? getEngineUptime(pid!) : null,
        logs: getLogTail(15),
    });
}

// ─── POST: Start / Stop engine ──────────────────────────────────────────────

export async function POST(request: Request) {
    try {
        const { action } = await request.json();

        if (action === 'start') {
            // Check if already running
            const existingPid = getStoredPid();
            if (existingPid !== null) {
                return NextResponse.json({
                    status: 'running',
                    pid: existingPid,
                    message: 'Engine is already running',
                });
            }

            // Ensure data dir exists
            if (!fs.existsSync(DATA_DIR)) {
                fs.mkdirSync(DATA_DIR, { recursive: true });
            }

            const pythonBin = getPythonBin();
            const mainPy = path.join(PROJECT_ROOT, 'main.py');

            if (!fs.existsSync(mainPy)) {
                return NextResponse.json(
                    { error: `main.py not found at ${mainPy}` },
                    { status: 404 }
                );
            }

            // Open log file for output
            const logFd = fs.openSync(LOG_FILE, 'a');

            // Spawn detached process
            const child = spawn(pythonBin, [mainPy], {
                cwd: PROJECT_ROOT,
                detached: true,
                stdio: ['ignore', logFd, logFd],
                env: { ...process.env },
            });

            const pid = child.pid!;
            child.unref();
            fs.closeSync(logFd);

            // Save PID
            fs.writeFileSync(PID_FILE, String(pid));

            console.log(`[engine] Started main.py with PID ${pid}`);

            return NextResponse.json({
                status: 'running',
                pid,
                message: `Engine started (PID ${pid})`,
            });

        } else if (action === 'stop') {
            const pid = getStoredPid();
            if (pid === null) {
                return NextResponse.json({
                    status: 'stopped',
                    message: 'Engine is not running',
                });
            }

            // Graceful shutdown: SIGTERM first
            try {
                process.kill(pid, 'SIGTERM');
            } catch { /* already gone */ }

            // Wait up to 3s, then SIGKILL
            await new Promise<void>((resolve) => {
                let checks = 0;
                const interval = setInterval(() => {
                    checks++;
                    if (!isProcessRunning(pid) || checks >= 6) {
                        clearInterval(interval);
                        if (isProcessRunning(pid)) {
                            try { process.kill(pid, 'SIGKILL'); } catch { /* ignore */ }
                        }
                        resolve();
                    }
                }, 500);
            });

            // Clean up PID file
            try { fs.unlinkSync(PID_FILE); } catch { /* ignore */ }

            console.log(`[engine] Stopped engine (PID ${pid})`);

            return NextResponse.json({
                status: 'stopped',
                message: `Engine stopped (PID ${pid})`,
            });

        } else {
            return NextResponse.json(
                { error: `Unknown action: ${action}. Use 'start' or 'stop'.` },
                { status: 400 }
            );
        }
    } catch (error: any) {
        console.error('[engine] Error:', error);
        return NextResponse.json(
            { error: error.message || 'Engine control failed' },
            { status: 500 }
        );
    }
}
