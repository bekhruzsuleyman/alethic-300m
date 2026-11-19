// All the screen's state in one pure function, so it can be tested without a terminal.
export const initial = () => ({
  items: [], nextId: 1, live: '', plan: null, code: false, pendingHead: null, busy: false, status: null,
  details: true, confirm: null, hello: null, clearKey: 0, quit: false,
})

const add = (s, item) => ({ ...s, items: [...s.items, { id: s.nextId, ...item }], nextId: s.nextId + 1 })

// Move text and plan that are still being streamed into the finished items.
export function flush(s) {
  if (s.live) s = add({ ...s, live: '' }, { kind: 'text', text: s.live })
  if (s.plan) s = add({ ...s, plan: null }, { kind: 'plan', label: s.plan.label, text: s.plan.text })
  return s
}

export function reduce(s, e) {
  switch (e.t) {
    case 'hello':
      return add({ ...s, hello: e }, { kind: 'banner', title: e.title, folder: e.folder, effort: e.effort, auto: e.auto })
    case 'text': {
      if (s.plan) s = flush(s)
      const buf = s.live + e.text
      const cut = buf.lastIndexOf('\n')
      if (cut < 0) return { ...s, live: buf }
      let out = { ...s, live: buf.slice(cut + 1) }
      for (const line of buf.slice(0, cut).split('\n')) out = add(out, { kind: 'text', text: line })
      return out
    }
    case 'plan':
      s = s.live ? flush(s) : s
      return { ...s, plan: s.plan && s.plan.label === e.kind ? { ...s.plan, text: s.plan.text + e.text } : { label: e.kind, text: e.text.replace(/^ +/, '') } }
    case 'code_open':
      return add({ ...flush(s), code: true }, { kind: 'codeTitle', lang: e.lang })
    case 'code':
      return add(s, { kind: 'code', runs: e.runs })
    case 'code_close':
      return add({ ...s, code: false }, { kind: 'codeEnd' })
    case 'call':
      return { ...flush(s), pendingHead: e.head }
    case 'tool':
      return add({ ...s, pendingHead: null }, { kind: 'tool', head: e.head, title: e.title, rows: e.rows, ok: e.ok, summary: e.summary, open: s.details })
    case 'status':
      return { ...s, status: e }
    case 'info':
      return add(flush(s), { kind: 'info', text: e.text, level: e.kind })
    case 'details':
      return { ...s, details: e.value }
    case 'doc':
      return add(s, { kind: 'doc', name: e.name, rows: e.rows })
    case 'confirm':
      return { ...flush(s), confirm: e }
    case 'clear':
      return { ...initial(), hello: s.hello, details: s.details, clearKey: s.clearKey + 1, nextId: s.nextId }
    case 'done': {
      s = flush(s)
      if (e.note) s = add(s, { kind: 'info', text: e.note, level: 'warn' })
      if (!e.command) s = add(s, { kind: 'stats', calls: e.calls, tokens: e.tokens, seconds: e.seconds, rate: e.rate })
      const hello = s.hello && e.effort ? { ...s.hello, effort: e.effort, auto: e.auto } : s.hello
      return { ...s, hello, busy: false, status: null, pendingHead: null, confirm: null }
    }
    case 'quit':
      return { ...s, quit: true }
    default:
      return s
  }
}

// Things the user does.
export const submit = (s, text) => add({ ...s, busy: true, status: null }, { kind: 'user', text })
export const answered = (s) => ({ ...s, confirm: null })
export const toggleDetails = (s) => ({ ...s, details: !s.details })
export function statsLine({ calls, tokens, seconds, rate }) {
  const n = (v, w) => `${v} ${w}${v === 1 ? '' : 's'}`
  return `${n(calls, 'tool call')} · ${n(tokens, 'token')} · ${seconds.toFixed(1)}s · ${rate.toFixed(1)} tok/s`
}
