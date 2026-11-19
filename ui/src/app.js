import React, { useEffect, useReducer, useRef, useState } from 'react'
import { Box, Static, Text, useApp, useInput, useStdout } from 'ink'
import { initial, reduce, submit, answered, toggleDetails, statsLine } from './state.js'
import { suggest } from './menu.js'
import { Item, Runs } from './view.js'
import { AMBER, CREAM, MUTED } from './style.js'

const h = React.createElement
const SPIN = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

function reducer(s, a) {
  if (a.type === 'event') return reduce(s, a.event)
  if (a.type === 'submit') return submit(s, a.text)
  if (a.type === 'answered') return answered(s)
  if (a.type === 'fold') return toggleDetails(s)
  return s
}

export function App({ backend }) {
  const { exit } = useApp()
  const { stdout } = useStdout()
  const [s, dispatch] = useReducer(reducer, undefined, initial)
  // the line being typed lives in a ref so two keys in one tick (fast typing, paste) both land
  const ed = useRef({ text: '', cursor: 0, pick: 0 })
  const [, redraw] = useState(0)
  const { text, cursor, pick } = ed.current
  const setPick = (n) => { ed.current.pick = n; redraw((x) => x + 1) }
  const setText = (v) => { ed.current.text = v; redraw((x) => x + 1) }
  const setCursor = (c) => { ed.current.cursor = c; redraw((x) => x + 1) }
  const [tick, setTick] = useState(0)
  const hist = useRef({ list: [], pos: 0 })

  useEffect(() => {
    const on = (event) => dispatch({ type: 'event', event })
    backend.on('event', on)
    return () => backend.off('event', on)
  }, [backend])
  useEffect(() => { if (s.quit) exit() }, [s.quit])
  useEffect(() => { if (s.clearKey) stdout.write('\x1b[2J\x1b[3J\x1b[H') }, [s.clearKey])
  useEffect(() => {
    if (!s.busy) return undefined
    const id = setInterval(() => setTick((t) => t + 1), 100)
    return () => clearInterval(id)
  }, [s.busy])

  const menu = !s.busy && s.hello ? suggest(s.hello.commands, text) : []
  const sel = Math.min(pick, Math.max(0, menu.length - 1))
  const set = (v, c = v.length) => { ed.current = { text: v, cursor: c, pick: 0 }; redraw((x) => x + 1) }

  useInput((input, key) => {
    if (key.ctrl && input === 'c') { backend.close(); exit(); return }
    if (key.ctrl && input === 'o') { dispatch({ type: 'fold' }); return }
    if (s.confirm) {
      const a = input.toLowerCase()
      if (a === 'y' || a === 'n' || a === 'a') { backend.send({ t: 'confirm', answer: a }); dispatch({ type: 'answered' }) }
      return
    }
    if (s.busy) return
    const { text, cursor, pick } = ed.current
    const menu = suggest(s.hello ? s.hello.commands : [], text)
    const sel = Math.min(pick, Math.max(0, menu.length - 1))
    if (key.return) {
      let v = text.trim()
      if (menu.length && v !== '/' + menu[sel].name && !s.hello.commands.some((c) => '/' + c.name === v)) { set('/' + menu[sel].name + ' '); return }
      if (!v) return
      hist.current = { list: [...hist.current.list, v], pos: hist.current.list.length + 1 }
      dispatch({ type: 'submit', text: v })
      backend.send({ t: 'submit', text: v })
      set('')
    } else if (key.tab) { if (menu.length) set('/' + menu[sel].name + ' ') }
    else if (key.upArrow || key.downArrow) {
      if (menu.length) setPick((sel + (key.upArrow ? menu.length - 1 : 1)) % menu.length)
      else {
        const hs = hist.current
        hs.pos = key.upArrow ? Math.max(0, hs.pos - 1) : Math.min(hs.list.length, hs.pos + 1)
        set(hs.list[hs.pos] || '')
      }
    } else if (key.leftArrow) setCursor(Math.max(0, cursor - 1))
    else if (key.rightArrow) setCursor(Math.min(text.length, cursor + 1))
    else if (key.backspace || key.delete) {
      if (cursor > 0) set(text.slice(0, cursor - 1) + text.slice(cursor), cursor - 1)
    } else if (input && !key.ctrl && !key.meta) {
      const add = input.replace(/[\r\n]+/g, ' ')
      set(text.slice(0, cursor) + add + text.slice(cursor), cursor + add.length)
    }
  })

  const st = s.status
  const before = text.slice(0, cursor), at = text.slice(cursor, cursor + 1) || ' ', after = text.slice(cursor + 1)
  return h(Box, { flexDirection: 'column' },
    h(Static, { items: s.items, key: s.clearKey }, (item) => h(Item, { key: item.id, item })),
    s.live ? h(Text, { color: CREAM }, s.live) : null,
    s.plan ? h(Text, { color: MUTED, italic: true }, '▸ ' + s.plan.label + ' ' + s.plan.text) : null,
    s.pendingHead ? h(Text, null, h(Text, { color: AMBER }, '▸ '), h(Text, { bold: true, color: CREAM }, s.pendingHead), h(Text, { color: MUTED }, ' …')) : null,
    s.confirm ? h(Box, { flexDirection: 'column', borderStyle: 'round', borderColor: AMBER, paddingX: 1 },
      h(Text, { bold: true, color: AMBER }, 'Run this command?'), h(Text, null, '$ ', h(Runs, { runs: s.confirm.cmd })),
      h(Text, { color: MUTED }, '[y]es   [n]o   [a]lways (do not ask again)')) : null,
    menu.length ? h(Box, { flexDirection: 'column', paddingX: 1 }, menu.slice(0, 8).map((c, i) =>
      h(Text, { key: c.name, color: i === sel ? AMBER : MUTED, bold: i === sel }, (i === sel ? '› ' : '  ') + c.usage.padEnd(20) + c.description))) : null,
    s.busy && st ? h(Text, { color: AMBER }, `${SPIN[tick % SPIN.length]} ${st.word}…  ${statsLine(st)}`) : s.busy ? h(Text, { color: AMBER }, `${SPIN[tick % SPIN.length]} …`) : null,
    s.busy && st ? h(Text, { color: MUTED }, 'tip: ' + st.tip) : null,
    h(Box, { borderStyle: 'round', borderColor: s.busy ? MUTED : AMBER, paddingX: 1 },
      s.busy ? h(Text, { color: MUTED }, text || 'working… (input is locked until the answer is done)')
        : text ? h(Text, null, before, h(Text, { inverse: true }, at), after)
          : h(Text, { color: MUTED }, h(Text, { inverse: true }, ' '), 'message the model, or type / for commands')),
    s.hello ? h(Text, { color: MUTED }, ` ${s.hello.effort} · ${s.hello.auto ? 'auto' : 'asks first'}`) : null)
}
