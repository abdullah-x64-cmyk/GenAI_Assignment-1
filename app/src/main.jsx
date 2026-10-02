import { useState } from 'react'
import {
  Activity,
  ArrowUpRight,
  BrainCircuit,
  ChevronDown,
  Clock3,
  Cpu,
  FileImage,
  Gauge,
  Layers3,
  Menu,
  Moon,
  Palette,
  ScanLine,
  Sparkles,
  Upload,
  X,
} from 'lucide-react'
import './index.css'

const workspaces = [
  { id: 'home', label: 'Home', icon: Gauge, accent: 'amber' },
  { id: 'universal', label: 'Universal Restoration', short: 'Universal', icon: Layers3, accent: 'teal' },
  { id: 'hard', label: 'Hard-Routed Restoration', short: 'Hard-Routed', icon: ScanLine, accent: 'blue' },
  { id: 'soft', label: 'Soft MoE Restoration', short: 'Soft MoE', icon: BrainCircuit, accent: 'violet' },
  { id: 'sketch', label: 'Face-to-Sketch', short: 'Face-to-Sketch', icon: Palette, accent: 'coral' },
]

const models = [
  { name: 'Universal', detail: 'multi-corruption denoiser', state: 'Ready', color: 'bg-teal-500' },
  { name: 'Hard Route', detail: 'classifier + specialists', state: 'Ready', color: 'bg-blue-500' },
  { name: 'Soft MoE', detail: 'continuous expert blend', state: 'Ready', color: 'bg-violet-500' },
  { name: 'Face-to-Sketch', detail: 'conditional GAN', state: 'Ready', color: 'bg-orange-500' },
]

function App() {
  const [active, setActive] = useState('home')
  const [mobileOpen, setMobileOpen] = useState(false)
  const activeWorkspace = workspaces.find((workspace) => workspace.id === active)

  const navigate = (id) => {
    setActive(id)
    setMobileOpen(false)
  }

  return (
    <div className="min-h-screen bg-[#f4f1e9] text-[#1e2928]">
      <div className="grain" />
      <header className="fixed inset-x-0 top-0 z-40 flex h-16 items-center justify-between border-b border-[#d9d4c8] bg-[#f4f1e9]/90 px-5 backdrop-blur-xl lg:hidden">
        <Brand />
        <button className="icon-button" onClick={() => setMobileOpen(!mobileOpen)} aria-label="Toggle navigation">
          {mobileOpen ? <X size={19} /> : <Menu size={19} />}
        </button>
      </header>
      {mobileOpen && <MobileNav active={active} navigate={navigate} />}
      <div className="flex min-h-screen">
        <aside className="sidebar hidden w-[276px] shrink-0 flex-col border-r border-[#d9d4c8] bg-[#ebe8de] px-5 py-6 lg:flex">
          <Brand />
          <div className="mt-12">
            <p className="eyebrow px-3">Workspaces</p>
            <nav className="mt-3 space-y-1">
              {workspaces.map((workspace) => <NavItem key={workspace.id} workspace={workspace} active={active} navigate={navigate} />)}
            </nav>
          </div>
          <div className="mt-auto space-y-5">
            <div className="status-card">
              <div className="flex items-center justify-between">
                <span className="eyebrow">Runtime</span>
                <span className="live-dot"><span /></span>
              </div>
              <p className="mt-3 text-sm font-semibold">ONNX Runtime</p>
              <p className="mt-1 text-xs text-[#68736e]">CPU execution provider</p>
            </div>
            <div className="flex items-center justify-between px-3 text-xs text-[#7b827c]">
              <span>Studio v0.1</span>
              <Moon size={14} />
            </div>
          </div>
        </aside>
        <main className="relative flex-1 overflow-hidden pt-16 lg:pt-0">
          <div className="mx-auto max-w-[1440px] px-5 py-7 sm:px-8 lg:px-12 lg:py-10">
            <Topbar workspace={activeWorkspace} />
            {active === 'home' ? <Home navigate={navigate} /> : <Workspace workspace={activeWorkspace} />}
          </div>
        </main>
      </div>
    </div>
  )
}

function Brand() {
  return <div className="flex items-center gap-3 px-3"><div className="brand-mark"><Sparkles size={17} strokeWidth={2.5} /></div><div><p className="brand-name">Restoration</p><p className="brand-sub">studio / 01</p></div></div>
}

function NavItem({ workspace, active, navigate }) {
  const Icon = workspace.icon
  const isActive = active === workspace.id
  return <button onClick={() => navigate(workspace.id)} className={`nav-item ${isActive ? 'nav-active' : ''}`}><Icon size={17} /><span>{workspace.label}</span>{isActive && <ArrowUpRight className="ml-auto" size={15} />}</button>
}

function MobileNav({ active, navigate }) {
  return <div className="fixed inset-x-0 top-16 z-30 border-b border-[#d9d4c8] bg-[#ebe8de] p-4 shadow-xl lg:hidden"><nav className="space-y-1">{workspaces.map((workspace) => <NavItem key={workspace.id} workspace={workspace} active={active} navigate={navigate} />)}</nav></div>
}

function Topbar({ workspace }) {
  return <div className="mb-10 flex items-end justify-between gap-5"><div><p className="eyebrow">Workspace / {workspace.id}</p><h1 className="display-title mt-2">{workspace.label}</h1></div><div className="hidden items-center gap-2 sm:flex"><span className="model-chip"><span className="h-2 w-2 rounded-full bg-emerald-500" /> Inference online</span><button className="icon-button"><Activity size={17} /></button></div></div>
}

function Home({ navigate }) {
  return <div className="animate-rise"><section className="home-hero"><div className="relative z-10 max-w-3xl"><p className="eyebrow text-[#cc6d43]">Image restoration / synthesis console</p><h2 className="hero-title mt-5">Make the image<br /><em>whole again.</em></h2><p className="hero-copy mt-6 max-w-xl">A focused lab for comparing universal restoration, routed experts, soft mixtures, and style-conditioned sketch generation.</p><div className="mt-8 flex flex-wrap gap-3"><button className="primary-button" onClick={() => navigate('universal')}>Open restoration lab <ArrowUpRight size={16} /></button><button className="secondary-button" onClick={() => navigate('sketch')}>Try face-to-sketch</button></div></div><div className="hero-orbit orbit-one" /><div className="hero-orbit orbit-two" /><div className="hero-grid" /></section><section className="mt-10"><div className="mb-4 flex items-center justify-between"><div><p className="eyebrow">Model registry</p><h3 className="section-title mt-1">Four ways to see the signal</h3></div><span className="text-xs text-[#7b827c]">all systems ready</span></div><div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">{models.map((model) => <ModelCard key={model.name} model={model} />)}</div></section><section className="mt-10 grid gap-3 lg:grid-cols-[1.5fr_1fr]"><div className="info-band"><div className="flex items-center gap-3"><div className="soft-icon bg-[#d9eee6] text-[#277963]"><FileImage size={18} /></div><div><p className="eyebrow">Latest evaluation</p><p className="mt-1 text-sm font-semibold">Oxford-IIIT Pet / 12 visual cases</p></div></div><span className="text-xs text-[#7b827c]">Task 1</span></div><div className="info-band"><div className="flex items-center gap-3"><div className="soft-icon bg-[#f5dfd5] text-[#bd5e39]"><Clock3 size={18} /></div><div><p className="eyebrow">Average inference</p><p className="mt-1 text-sm font-semibold">48 ms / image</p></div></div><span className="text-xs text-[#7b827c]">CPU</span></div></section></div>
}

function ModelCard({ model }) {
  return <article className="model-card"><div className="flex items-start justify-between"><span className={`h-2.5 w-2.5 rounded-full ${model.color}`} /><span className="ready-pill">{model.state}</span></div><h4 className="mt-9 text-lg font-semibold tracking-[-0.02em]">{model.name}</h4><p className="mt-1 text-sm text-[#68736e]">{model.detail}</p><div className="mt-6 flex items-center gap-2 text-xs text-[#8a918a]"><Cpu size={13} /> ONNX / CPU</div></article>
}

function Workspace({ workspace }) {
  const Icon = workspace.icon
  const isSketch = workspace.id === 'sketch'
  return <div className="animate-rise"><div className="workspace-banner"><div className={`workspace-symbol accent-${workspace.accent}`}><Icon size={21} /></div><div><p className="eyebrow">{isSketch ? 'Conditional generation' : 'Restoration pipeline'}</p><h2 className="mt-1 text-xl font-semibold tracking-[-0.025em]">{workspace.label}</h2></div><span className="ml-auto hidden text-xs text-[#7b827c] md:block">{isSketch ? '3 style conditions' : '128 × 128 input'}</span></div><div className="mt-5 grid gap-5 xl:grid-cols-[1fr_300px]"><div className="canvas-panel"><div className="flex items-center justify-between border-b border-[#ded9ce] px-5 py-4"><div className="flex items-center gap-2 text-sm font-semibold"><span className="h-2 w-2 rounded-full bg-[#cc6d43]" />Input canvas</div><span className="text-xs text-[#87908a]">Awaiting image</span></div><div className="empty-canvas"><div className="upload-circle"><Upload size={20} /></div><p className="mt-4 text-sm font-semibold">Drop an image to begin</p><p className="mt-1 text-xs text-[#87908a]">PNG or JPG · up to 10 MB</p><button className="secondary-button mt-5"><Upload size={15} /> Select image</button></div></div><ControlPanel workspace={workspace} isSketch={isSketch} /></div></div>
}

function ControlPanel({ workspace, isSketch }) {
  return <aside className="control-panel"><div className="flex items-center gap-2"><div className={`mini-symbol accent-${workspace.accent}`}><workspace.icon size={15} /></div><span className="text-sm font-semibold">Controls</span></div><div className="mt-7 space-y-6"><label className="control-label">{isSketch ? 'Sketch style' : 'Corruption mode'}<div className="select-wrap"><select defaultValue={isSketch ? 'Style 01' : workspace.id === 'hard' ? 'Auto-detect' : 'Clean sample'}><option>Style 01</option><option>Style 02</option><option>Style 03</option><option>Auto-detect</option><option>Clean sample</option></select><ChevronDown size={15} /></div></label>{!isSketch && <label className="control-label">Severity<div className="segmented"><button className="selected">Low</button><button>Med</button><button>High</button></div></label>}{workspace.id === 'soft' && <div><div className="mb-2 flex justify-between"><span className="control-label">Temperature</span><span className="text-xs text-[#277963]">0.78</span></div><input className="w-full accent-[#277963]" type="range" min="0.25" max="2" step="0.01" defaultValue="0.78" /></div>}<button className="primary-button w-full justify-center">Run inference <ArrowUpRight size={15} /></button></div><div className="mt-12 border-t border-[#ded9ce] pt-5"><div className="flex items-center justify-between text-xs"><span className="text-[#87908a]">Model status</span><span className="flex items-center gap-1.5 text-[#277963]"><span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />Ready</span></div><div className="mt-3 flex items-center justify-between text-xs"><span className="text-[#87908a]">Expected time</span><span className="font-semibold">~48 ms</span></div></div></aside>
}

export default App
