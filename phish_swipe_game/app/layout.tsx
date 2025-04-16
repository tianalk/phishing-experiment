import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Phish Swipe - AI vs. Human Detection Research',
  description: 'Test your ability to identify phishing emails as part of our research comparing human-generated and AI-generated phishing attempts.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
} 