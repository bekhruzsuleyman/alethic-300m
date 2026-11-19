import { EventEmitter } from 'node:events'
import { spawn } from 'node:child_process'
import readline from 'node:readline'

// The Python agent as a child process: JSON lines in and out.
export class Backend extends EventEmitter {
  constructor(command, args, options = {}) {
    super()
    this.child = spawn(command, args, { stdio: ['pipe', 'pipe', 'inherit'], ...options })
    readline.createInterface({ input: this.child.stdout }).on('line', (line) => {
      try { this.emit('event', JSON.parse(line)) } catch { /* not ours: ignore */ }
    })
    this.child.on('exit', (code) => this.emit('exit', code))
  }
  send(msg) { this.child.stdin.write(JSON.stringify(msg) + '\n') }
  close() { try { this.child.stdin.end(); this.child.kill() } catch { /* already gone */ } }
}
