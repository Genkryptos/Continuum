import { Hero } from './components/Hero';
import { Features } from './components/Features';
import { Architecture } from './components/Architecture';
import { Roadmap } from './components/Roadmap';

function App() {
  return (
    <div className="app-container">
      <header className="navbar">
        <div className="nav-brand">
          <div className="brand-icon"></div>
          Continuum
        </div>
        <nav className="nav-links">
          <a href="#features">Features</a>
          <a href="#architecture">Architecture</a>
          <a href="#roadmap">Roadmap</a>
        </nav>
      </header>

      <main className="main-content">
        <Hero />
        <Features />
        <Architecture />
        <Roadmap />
      </main>

      <footer className="footer section text-center">
        <div className="container">
          <div className="brand-icon mx-auto mb-4" style={{opacity: 0.8}}></div>
          <h2 className="heading-2 mb-4">Ready to bypass context limits?</h2>
          <p className="subtitle mx-auto mb-8">
            Continuum is openly built structually sound to support whatever agent orchestrators need regarding memory caching.
          </p>
          <p className="text-secondary text-sm">
            &copy; {new Date().getFullYear()} Continuum Architecture / Open Source Setup
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
