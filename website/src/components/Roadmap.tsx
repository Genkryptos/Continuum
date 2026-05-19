import { CheckCircle2, CircleDashed, Clock } from 'lucide-react';

export function Roadmap() {
  const phases = [
    {
      phase: "Phase 1: Foundation",
      title: "Hot & Warm Memory Loop",
      description: "Establish the underlying STM and MTM pipelines with automated token compression, PostgreSQL+pgvector storage, and OpenAI embeddings.",
      status: "completed",
      icon: <CheckCircle2 color="var(--brand-primary)" size={24} />
    },
    {
      phase: "Phase 2: Externalization",
      title: "API Framework Abstraction",
      description: "Bridge the experimental loops into a robust FastAPI interface, enabling any agent, regardless of its core logic, to inject Continuum memory capabilities.",
      status: "progress",
      icon: <Clock color="var(--text-tertiary)" size={24} />
    },
    {
      phase: "Phase 3: The Deep Archive",
      title: "Long-Term Memory (LTM)",
      description: "Expand the pipeline to index, organize, and recall deep archival knowledge mapping connections across infinite context windows.",
      status: "planned",
      icon: <CircleDashed color="var(--text-secondary)" size={24} />
    },
    {
      phase: "Phase 4: Evolution",
      title: "Advanced Memory Schemas",
      description: "Scale beyond naive LLM summaries into knowledge-graph compilation, dynamic retrieval algorithms, and autonomous structural adaptation.",
      status: "planned",
      icon: <CircleDashed color="var(--text-secondary)" size={24} />
    }
  ];

  return (
    <section id="roadmap" className="section container">
      <div className="text-center mb-16">
        <h2 className="heading-2 mb-4">Strategic Timeline</h2>
        <p className="subtitle mx-auto">
          Continuum is being built to withstand paradigm shifts. We are solving context limits today while architecting for the algorithms of tomorrow.
        </p>
      </div>

      <div className="max-w-4xl mx-auto">
        <div className="flex flex-col gap-6">
          {phases.map((item, idx) => (
            <div key={idx} className="glass-panel flex gap-6 items-start relative overflow-hidden">
              {item.status === 'completed' && <div className="absolute top-0 left-0 w-1 h-full shadow-[0_0_10px_rgba(0,212,255,0.8)]" style={{background: 'var(--brand-primary)'}} />}
              {item.status === 'progress' && <div className="absolute top-0 left-0 w-1 h-full shadow-[0_0_10px_rgba(74,111,165,0.8)]" style={{background: 'var(--text-tertiary)'}} />}
              {item.status === 'planned' && <div className="absolute top-0 left-0 w-1 h-full opacity-50" style={{background: 'var(--text-secondary)'}} />}
              
              <div className="mt-1 p-3 rounded-full border" style={{background: 'rgba(0, 0, 0, 0.4)', borderColor: 'var(--border-subtle)'}}>
                {item.icon}
              </div>
              
              <div>
                <p className="text-secondary mb-1" style={{fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.1em'}}>
                  {item.phase.toUpperCase()}
                </p>
                <h3 className="heading-3 mb-2" style={{color: '#fff', textTransform: 'none'}}>{item.title}</h3>
                <p className="text-secondary">{item.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
