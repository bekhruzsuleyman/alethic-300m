#!/usr/bin/env node
// alethic-agent screen: node ui/index.js <python> [args for alethic.agent...]
import React from 'react'
import { render } from 'ink'
import { App } from './src/app.js'
import { Backend } from './src/backend.js'

const [python, ...args] = process.argv.slice(2)
if (!python) { console.error('usage: node index.js <python> --weights ... --tokenizer ...'); process.exit(2) }
const backend = new Backend(python, ['-m', process.env.ALETHIC_BACKEND_MODULE || 'alethic.agent', '--json', ...args])
const app = render(React.createElement(App, { backend }))
backend.on('exit', (code) => { app.unmount(); process.exit(code || 0) })
app.waitUntilExit().then(() => { backend.close(); process.exit(0) })
