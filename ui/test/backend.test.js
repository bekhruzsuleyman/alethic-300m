import test from 'node:test'
import assert from 'node:assert/strict'
import { Backend } from '../src/backend.js'

test('the backend turns JSON lines from the child into events and ignores other lines', async () => {
  const script = "import sys\nprint('not json')\nprint('{\"t\": \"hello\", \"title\": \"T\"}')\nfor l in sys.stdin: print(l.strip())"
  const b = new Backend('python3', ['-c', script])
  const got = []
  b.on('event', (e) => got.push(e))
  b.send({ t: 'submit', text: 'hi' })
  await new Promise((r) => setTimeout(r, 400))
  b.close()
  assert.deepEqual(got, [{ t: 'hello', title: 'T' }, { t: 'submit', text: 'hi' }])
})
