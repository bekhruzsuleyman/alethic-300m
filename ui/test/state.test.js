import test from 'node:test'
import assert from 'node:assert/strict'
import { initial, reduce, submit, statsLine } from '../src/state.js'

const run = (events, s = initial()) => events.reduce(reduce, s)

test('text becomes finished lines as soon as a line is complete, the rest stays live', () => {
  const s = run([{ t: 'text', text: 'Hello wor' }, { t: 'text', text: 'ld.\nSecond' }])
  assert.deepEqual(s.items.map((i) => i.text), ['Hello world.'])
  assert.equal(s.live, 'Second')
})

test('done flushes the last partial line and ends the busy state', () => {
  let s = submit(initial(), 'hi')
  assert.equal(s.busy, true)
  s = run([{ t: 'text', text: 'Bye' }, { t: 'done', note: '', calls: 0, tokens: 2, seconds: 1, rate: 2 }], s)
  assert.equal(s.busy, false)
  assert.deepEqual(s.items.map((i) => i.kind), ['user', 'text', 'stats'])
})

test('code events make a title, colored rows and an end', () => {
  const s = run([{ t: 'code_open', lang: 'python' }, { t: 'code', runs: [['def', '#7B95FF']] }, { t: 'code_close' }])
  assert.deepEqual(s.items.map((i) => i.kind), ['codeTitle', 'code', 'codeEnd'])
})

test('a tool result starts open or folded depending on the details setting', () => {
  const tool = { t: 'tool', head: 'Running command in terminal', title: 'bash', rows: [], ok: true, summary: 'hi' }
  assert.equal(run([tool]).items[0].open, true)
  assert.equal(run([{ t: 'details', value: false }, tool]).items[0].open, false)
})

test('a call shows as running until its result arrives', () => {
  let s = run([{ t: 'call', head: 'Writing a.py' }])
  assert.equal(s.pendingHead, 'Writing a.py')
  s = reduce(s, { t: 'tool', head: 'Writing a.py', title: 'a.py', rows: [], ok: true, summary: 'ok' })
  assert.equal(s.pendingHead, null)
})

test('clear empties everything and changes the clear key so the screen is rebuilt', () => {
  const s = run([{ t: 'hello', title: 'T', folder: 'f', effort: 'medium', auto: true, commands: [] }, { t: 'text', text: 'x\n' }, { t: 'clear' }])
  assert.equal(s.items.length, 0)
  assert.equal(s.clearKey, 1)
  assert.equal(s.hello.title, 'T')
})

test('plan pieces join into one plan item', () => {
  const s = run([{ t: 'plan', kind: 'plan', text: ' one ' }, { t: 'plan', kind: 'plan', text: 'two' }, { t: 'text', text: 'x\n' }])
  assert.equal(s.items[0].kind, 'plan')
  assert.equal(s.items[0].text, 'one two')
})

test('the stats line reads like the classic one', () => {
  assert.equal(statsLine({ calls: 1, tokens: 2, seconds: 1.25, rate: 3 }), '1 tool call · 2 tokens · 1.3s · 3.0 tok/s')
})
