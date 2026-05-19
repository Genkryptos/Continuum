import { BrainCircuit, ChevronRight, FileCode2 } from 'lucide-react';
import './Hero.css';

export function Hero() {
  return (
    <section className="hero-section section">
      <div className="hero-background">
        <div className="hero-grid"></div>
      </div>
      
      <div className="container hero-content text-center flex flex-col items-center justify-center animate-fade-in">
        <div className="badge badge-ready mb-6">
          <span className="badge-dot"></span>
          Agent-Centric Memory Engine
        </div>
        
        <h1 className="heading-1 mb-6">
          Intelligence with <br />
          <span className="text-gradient">Continuity.</span>
        </h1>
        
        <p className="subtitle mb-8">
          Continuum is a memory-first AI framework that provides seamless Short-, Mid-, and Long-Term Memory (STM/MTM/LTM) across sessions. It prevents context collapse and optimizes token usage via an intelligent retrieval and LLM-powered compression pipeline.
        </p>
        
        <div className="flex gap-4">
          <a href="#architecture" className="btn btn-primary">
            <BrainCircuit size={20} />
            Explore Architecture
          </a>
          <a href="#codebase" className="btn btn-secondary">
            <FileCode2 size={20} />
            View Documentation
            <ChevronRight size={16} />
          </a>
        </div>
      </div>
    </section>
  );
}
