import { Database, Zap, Globe, Layers } from 'lucide-react';

export function Features() {
  const features = [
    {
      icon: <Zap size={24} color="var(--brand-primary)" />,
      title: "Short-Term Memory (STM)",
      description: "Token-aware conversation history with automated, customizable token budgets and LLM-driven naive compression.",
      status: "ready",
      statusLabel: "Implemented"
    },
    {
      icon: <Database size={24} color="var(--brand-primary)" />,
      title: "Mid-Term Memory (MTM)",
      description: "Persistent Postgres + pgvector storage utilizing OpenAI embeddings and scored retrieval loops for continuous context grounding.",
      status: "ready",
      statusLabel: "Implemented"
    },
    {
      icon: <Globe size={24} color="var(--brand-primary)" />,
      title: "Live Knowledge Injection",
      description: "MCP-powered web search service built-in, injecting real-time data dynamically when local memory context is insufficient.",
      status: "ready",
      statusLabel: "Implemented"
    },
    {
      icon: <Layers size={24} color="var(--brand-primary)" />,
      title: "Plug-and-Play API Layer",
      description: "A FastAPI surface specifically designed to abstract memory complexities, letting any external agent framework hook in easily.",
      status: "progress",
      statusLabel: "In Progress"
    }
  ];

  return (
    <section id="features" className="section container">
      <div className="text-center mb-16">
        <h2 className="heading-2 mb-4">Core Capabilities</h2>
        <p className="subtitle mx-auto">
          Built upon robust interfaces to handle token eviction, dynamic retrieval, and context summarization.
        </p>
      </div>
      
      <div className="grid grid-cols-2">
        {features.map((pkg, idx) => (
          <div key={idx} className="glass-panel">
            <div className="flex justify-between items-start mb-6">
              <div className="feature-icon-wrapper" style={{background: 'rgba(0, 212, 255, 0.05)', padding: '0.75rem', borderRadius: '0.75rem', border: '1px solid rgba(0, 212, 255, 0.15)'}}>
                {pkg.icon}
              </div>
              <div className={`badge badge-${pkg.status}`}>
                <span className="badge-dot"></span>
                {pkg.statusLabel}
              </div>
            </div>
            <h3 className="heading-3 mb-3">{pkg.title}</h3>
            <p className="text-secondary">{pkg.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
