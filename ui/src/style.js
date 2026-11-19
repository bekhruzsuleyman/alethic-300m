// Our style names ("bold", "red", "#RRGGBB") as props for an Ink <Text>.
export const AMBER = '#FFB020', CREAM = '#F5E9C8', MUTED = '#A39E8A', BLOCK_BG = '#1C1B17'

const COLORS = { red: '#FF6B6B', green: '#9CCB8B', yellow: AMBER, blue: '#7B95FF', brand: AMBER, dim: MUTED, muted: MUTED }

export function styleProps(style) {
  const p = {}
  for (const part of String(style || '').split(' ').filter(Boolean)) {
    if (part === 'bold') p.bold = true
    else if (part === 'italic') p.italic = true
    else if (part.startsWith('#')) p.color = part
    else if (COLORS[part]) p.color = COLORS[part]
  }
  return p
}

export function runLength(runs) {
  return runs.reduce((n, [t]) => n + [...t].length, 0)
}

// Cut runs to `width` characters, ending with … when something was cut.
export function clipRuns(runs, width) {
  if (runLength(runs) <= width) return runs
  const out = []
  let left = Math.max(0, width - 1)
  for (const [t, s] of runs) {
    const chars = [...t]
    if (chars.length <= left) { out.push([t, s]); left -= chars.length; continue }
    out.push([chars.slice(0, left).join('') + '…', s])
    return out
  }
  return out
}
