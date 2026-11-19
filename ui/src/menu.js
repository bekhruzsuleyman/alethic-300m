export function suggest(commands, buffer) {
  if (!buffer.startsWith('/') || buffer.includes(' ')) return []
  const head = buffer.slice(1)
  return commands.filter((c) => c.name.startsWith(head))
}
