import GameContainer from '../components/GameContainer';
import styles from './page.module.css';
import { FaShieldAlt } from 'react-icons/fa';

export default function Home() {
  return (
    <main className={styles.main}>
      <div className={styles.contentWrapper}>
        <header className={styles.header}>
          <div className={styles.logoWrapper}>
            <FaShieldAlt className={styles.logoIcon} />
            <h1 className={styles.title}>
              Phish Swipe
            </h1>
          </div>
          <p className={styles.description}>
            Can you identify phishing emails? Swipe right on suspicious emails, left on legitimate ones.
          </p>
        </header>

        <GameContainer />

        <footer className={styles.footer}>
          <p className={styles.researchNote}>
            Part of a research project comparing human and AI phishing detection.
          </p>
          <p className={styles.copyright}>
            &copy; {new Date().getFullYear()} Phish Forensics Research
          </p>
        </footer>
      </div>
    </main>
  );
} 