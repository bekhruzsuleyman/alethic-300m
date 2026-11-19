import test from 'node:test'
import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import React from 'react'
import { render } from 'ink-testing-library'
import { App } from '../src/app.js'

const h = React.createElement
const live = []
test.afterEach(() => { while (live.length) live.pop().unmount() })
const wait = (ms = 60) => new Promise((r) => setTimeout(r, ms))

function fake() {
  const b = new EventEmitter()
  b.sent = []
  b.send = (m) => b.sent.push(m)
  b.close = () => {}
  b.emit_ = (e) => b.emit('event', e)
  return b
}
const HELLO = { t: 'hello', title: 'Alethic test', folder: '/work', effort: 'medium', auto: true,
  commands: [{ name: 'clear', usage: '/clear', description: 'like /new' }, { name: 'details', usage: '/details [on|off]', description: 'tool blocks' }] }

async function start() {
  const b = fake()
  const r = render(h(App, { backend: b }))
  await wait()
  b.emit_(HELLO)
  await wait()
  live.push(r)
  return { b, r }
}

test('the banner shows the title and the folder, and the prompt is ready', async () => {
  const { r } = await start()
  const f = r.lastFrame()
  assert.match(f, /Alethic test/)
  assert.match(f, /\/work/)
  assert.match(f, /message the model/)
})

test('typing and Enter sends the message and locks the input while it answers', async () => {
  const { b, r } = await start()
  r.stdin.write('hello')
  await wait()
  r.stdin.write('\r')
  await wait()
  assert.deepEqual(b.sent, [{ t: 'submit', text: 'hello' }])
  assert.match(r.lastFrame(), /input is locked/)
  r.stdin.write('x')
  await wait()
  assert.doesNotMatch(r.lastFrame(), /x\n/)
  b.emit_({ t: 'text', text: 'Hi there.\n' })
  b.emit_({ t: 'done', note: '', calls: 0, tokens: 3, seconds: 1, rate: 3 })
  await wait()
  assert.match(r.lastFrame(), /Hi there\./)
  assert.match(r.lastFrame(), /message the model/)
})

test('a tool block shows its header and body, and Ctrl+O folds the next ones', async () => {
  const { b, r } = await start()
  b.emit_({ t: 'tool', head: 'Running command in terminal', title: 'bash', rows: [[['$ echo hi', 'bold']], [['hi', '']]], ok: true, summary: 'hi' })
  await wait()
  assert.match(r.lastFrame(), /▾ Running command in terminal/)
  assert.match(r.lastFrame(), /\$ echo hi/)
  r.stdin.write('\x0f')
  await wait()
  b.emit_({ t: 'tool', head: 'Reading a.py lines 1-3', title: 'a.py', rows: [[['x = 1', '']]], ok: true, summary: 'x = 1' })
  await wait()
  assert.match(r.lastFrame(), /▸ Reading a\.py lines 1-3/)
  assert.doesNotMatch(r.lastFrame(), /x = 1\s*\n.*x = 1/s)
})

test('typing a slash lists the commands and Tab completes the first', async () => {
  const { r } = await start()
  r.stdin.write('/')
  r.stdin.write('c')
  await wait()
  assert.match(r.lastFrame(), /\/clear\s+like \/new/)
  r.stdin.write('\t')
  await wait()
  assert.match(r.lastFrame(), /\/clear /)
})

test('while working the status row and a tip are shown', async () => {
  const { b, r } = await start()
  r.stdin.write('hi')
  await wait()
  r.stdin.write('\r')
  await wait()
  b.emit_({ t: 'status', calls: 1, tokens: 10, seconds: 2, rate: 5, word: 'Pondering', tip: 'press Ctrl+O to fold blocks' })
  await wait()
  assert.match(r.lastFrame(), /Pondering…/)
  assert.match(r.lastFrame(), /tip: press Ctrl\+O/)
  assert.match(r.lastFrame(), /10 tokens/)
})

test('a command that needs approval asks, and y sends the answer', async () => {
  const { b, r } = await start()
  b.emit_({ t: 'confirm', cmd: [['cp', '#7B95FF'], [' a b', '']] })
  await wait()
  assert.match(r.lastFrame(), /Run this command\?/)
  r.stdin.write('y')
  await wait()
  assert.deepEqual(b.sent.at(-1), { t: 'confirm', answer: 'y' })
  assert.doesNotMatch(r.lastFrame(), /Run this command\?/)
})

test('/clear empties what was shown', async () => {
  const { b, r } = await start()
  b.emit_({ t: 'text', text: 'old answer\n' })
  await wait()
  assert.match(r.lastFrame(), /old answer/)
  b.emit_({ t: 'clear' })
  await wait()
  assert.doesNotMatch(r.lastFrame(), /old answer/)
})
