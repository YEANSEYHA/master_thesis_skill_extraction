import "./globals.css";

export const metadata = {
  title: "Curriculum alignment rating — ITC",
  description:
    "Expert rating study: how well does an ITC course outcome teach a skill demanded in Cambodian IT job postings?",
};

export const viewport = {
  themeColor: "#ffffff",
  colorScheme: "light",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
