export function Architecture() {
  return (
    <section id="architecture" className="section container">
      <div className="text-center mb-16">
        <h2 className="heading-2 mb-4">The Continuum Pipeline</h2>
        <p className="subtitle mx-auto">
          An extensible layout bridging transient context and vectorized permanence. Built with modularity to easily adopt future retrieval algorithms.
        </p>
      </div>
      
      <div className="glass-panel p-8">
        <div className="arch-flow-diagram">
          {/* Note: Ideally this would be the actual SVGs from docs, substituting with styled HTML to match UI */}
          <div className="flex flex-col md:flex-row gap-6 items-center w-full" style={{flexDirection: 'row', display: 'flex', justifyContent: 'center'}}>
            
            {/* API / Orchestration */}
            <div className="glass-panel text-center flex-1 w-full" style={{border: '1px solid rgba(0, 212, 255, 0.2)', boxShadow: '0 0 20px rgba(0, 212, 255, 0.05)'}}>
              <h4 className="heading-3 mb-2" style={{color: '#fff', fontSize: '1.25rem', letterSpacing: '0'}}>Agent Orchestrator</h4>
              <p className="text-sm text-secondary">LocalAgent / FastAPI</p>
            </div>
            
            <div className="text-gradient mx-2" style={{fontSize: '2rem'}}>↔</div>
            
            {/* Hot Memory */}
            <div className="glass-panel text-center flex-1 w-full relative" style={{border: '1px solid rgba(74, 111, 165, 0.3)', boxShadow: '0 0 20px rgba(74, 111, 165, 0.1)'}}>
              <div className="badge badge-ready mb-3 absolute" style={{top: '-12px', right: '-12px'}}>Hot</div>
              <h4 className="heading-3 mb-2" style={{color: '#fff', fontSize: '1.25rem', letterSpacing: '0'}}>STM Buffer</h4>
              <p className="text-sm text-secondary">Token Config & Compression</p>
            </div>
            
            <div className="text-gradient mx-2" style={{fontSize: '2rem'}}>↔</div>
            
            {/* Cold Memory */}
            <div className="glass-panel text-center flex-1 w-full relative" style={{border: '1px solid rgba(160, 174, 192, 0.2)'}}>
               <div className="badge badge-progress mb-3 absolute" style={{top: '-12px', right: '-12px'}}>Warm</div>
              <h4 className="heading-3 mb-2" style={{color: '#fff', fontSize: '1.25rem', letterSpacing: '0'}}>MTM Store</h4>
              <p className="text-sm text-secondary">pgvector / OpenAI Embeds</p>
            </div>
          </div>
          
          <div className="mt-12 text-center max-w-2xl mx-auto">
            <h3 className="heading-3 mb-4">Engineered for Extensibility</h3>
            <p className="text-secondary">
              Continuum separates the <code>STMCallbacks</code>, <code>MTMRepository</code>, and <code>LocalAgent</code> logic. This interface-first strategy ensures that whether you're using naive summarization today or advanced Graph-RAG tomorrow, the pipeline won't break.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
