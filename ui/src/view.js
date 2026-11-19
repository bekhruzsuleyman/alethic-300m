import React from 'react'
import { Box, Text } from 'ink'
import { AMBER, BLOCK_BG, CREAM, MUTED, clipRuns, runLength, styleProps } from './style.js'
import { statsLine } from './state.js'

const h = React.createElement
const width = () => Math.max(40, (process.stdout.columns || 100) - 2)

export function Runs({ runs }) {
  return runs.map(([t, s], i) => h(Text, { key: i, ...styleProps(s) }, t))
}

// One row of a block: colored runs on the block background, padded to the full width.
export function BlockRow({ runs, w = width(), indent = 0 }) {
  const room = w - indent - 2
  const clipped = clipRuns(runs, room)
  const pad = Math.max(0, room - runLength(clipped))
  return h(Text, null, ' '.repeat(indent), h(Text, { backgroundColor: BLOCK_BG }, ' ', h(Runs, { runs: clipped }), ' '.repeat(pad + 1)))
}

const MARK = [
  [0, 2, 0], [1, 2, 1], [2, 1, 2], [1, 2, 1], [0, 2, 0],
]
export function Logo() {
  // two overlapping discs: blue left, amber right, cream where they meet
  const rows = ['  ████ ████  ', ' █████████████', '███████████████', ' █████████████', '  ████ ████  ']
  const half = [2, 4, 5, 4, 2]
  return h(Box, { flexDirection: 'column', marginRight: 2 }, [0, 1, 2, 3, 4].map((y) => {
    const cells = []
    for (let x = 0; x < 17; x++) {
      const a = Math.abs(x - 5) <= half[y], b = Math.abs(x - 11) <= half[y]
      cells.push(h(Text, { key: x, color: a && b ? CREAM : a ? '#4F6BFF' : '#F2B01E' }, a || b ? '█' : ' '))
    }
    return h(Text, { key: y }, cells)
  }))
}

export function Item({ item }) {
  switch (item.kind) {
    case 'banner':
      return h(Box, { marginY: 1 }, h(Logo), h(Box, { flexDirection: 'column' },
        h(Text, { bold: true, color: AMBER }, item.title),
        h(Text, { color: CREAM }, 'folder  ' + item.folder),
        h(Text, { color: MUTED }, `${item.effort} · ${item.auto ? 'runs commands without asking' : 'asks before running commands'}`),
        h(Text, { color: MUTED }, 'type / for commands · Ctrl+O folds or opens tool blocks · the wheel scrolls')))
    case 'user':
      return h(Box, { marginTop: 1 }, h(Text, { color: AMBER, bold: true }, '› ' + item.text))
    case 'text':
      return h(Text, { color: CREAM }, item.text || ' ')
    case 'plan':
      return h(Text, { color: MUTED, italic: true }, '▸ ' + item.label + ' ' + item.text)
    case 'codeTitle':
      return h(Box, { marginTop: 1 }, h(BlockRow, { runs: [[item.lang, MUTED]] }))
    case 'code':
      return h(BlockRow, { runs: item.runs.length ? item.runs : [['', '']] })
    case 'codeEnd':
      return h(BlockRow, { runs: [['', '']] })
    case 'info':
      return h(Text, { color: item.level === 'warn' ? AMBER : MUTED }, item.text || ' ')
    case 'stats':
      return h(Box, { marginTop: 1 }, h(Text, { color: MUTED }, statsLine(item)))
    case 'tool':
      return h(Tool, { item })
    case 'doc':
      return h(Doc, { item })
    default:
      return null
  }
}

export function Tool({ item }) {
  const head = h(Text, null, h(Text, { color: AMBER }, item.open ? '▾ ' : '▸ '), h(Text, { bold: true, color: CREAM }, item.head),
    item.open ? null : h(Text, { color: item.ok ? MUTED : '#FF6B6B' }, '  · ' + item.summary))
  return h(Box, { flexDirection: 'column', marginTop: 1 }, head,
    item.open && item.rows.length ? [h(BlockRow, { key: 't', runs: [[item.title, MUTED]], indent: 2 }),
      ...item.rows.map((r, i) => h(BlockRow, { key: i, runs: r.length ? r : [['', '']], indent: 2 })),
      h(BlockRow, { key: 'e', runs: [['', '']], indent: 2 })] : null)
}

export function Doc({ item }) {
  const w = String(item.rows.length).length
  return h(Box, { flexDirection: 'column', marginTop: 1 },
    h(Text, { bold: true, color: AMBER }, `${item.name} · ${item.rows.length} lines`),
    item.rows.map((r, i) => h(BlockRow, { key: i, runs: [[String(i + 1).padStart(w + 1), MUTED], ['  ', ''], ...(r.length ? r : [['', '']])] })))
}
