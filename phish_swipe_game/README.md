# Phish Swipe Game

A web-based game for evaluating human ability to detect phishing emails. Part of the Phish Forensics research project comparing AI-generated vs. human-generated phishing emails.

## Features

- Swipe-based interface for classifying emails as "phishing" or "legitimate"
- Immediate feedback after each classification
- Progress tracking and scoring
- Data collection for research analysis
- Responsive design for both desktop and mobile

## Tech Stack

- **Frontend**: Next.js, React, TailwindCSS
- **Backend**: Supabase (PostgreSQL)
- **Data Collection**: CSV import script for email datasets

## Setup

### Prerequisites

- Node.js 18+ and npm
- A Supabase account and project

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd phish_swipe_game
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Create a `.env.local` file based on `.env.local.example`:
   ```
   cp .env.local.example .env.local
   ```

4. Update the `.env.local` file with your Supabase credentials:
   ```
   NEXT_PUBLIC_SUPABASE_URL=your-supabase-project-url
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
   SUPABASE_SERVICE_KEY=your-supabase-service-role-key
   ```

### Database Setup

1. In your Supabase project, navigate to the SQL Editor.

2. Run the SQL commands from `supabase/schema.sql` to create the required tables.

### Data Import

1. Prepare your CSV files for human-generated and AI-generated emails. Each CSV should have columns:
   - `Email_ID`
   - `Email_Body`
   - `True_Label` (values should be either 'phishing' or 'legitimate')

2. Install additional dependencies for the import script:
   ```
   npm install -D csv-parser dotenv minimist
   ```

3. Run the import script:
   ```
   npm run import-data -- --human=path/to/human_emails.csv --ai=path/to/ai_emails.csv
   ```

   Add the `--clear` flag to remove existing data before import.

## Development

Run the development server:

```
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Deployment

1. Build the application:
   ```
   npm run build
   ```

2. Deploy to your preferred hosting platform (Vercel, Netlify, etc.):
   ```
   npm run start
   ```

## Research Usage

Data collected from user classifications is stored in the `user_classifications` table in Supabase. You can:

1. Export this data for analysis
2. Use SQL queries to calculate metrics (example queries in `supabase/schema.sql`)
3. Compare human detection rates between AI-generated and human-generated phishing emails

## License

[MIT License](LICENSE)
