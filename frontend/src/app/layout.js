import "./globals.css";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Job Site Discovery",
  description: "Discover local job portals across countries",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      {/* suppressHydrationWarning: a browser extension injects attributes
          (bis_register, __processed_<uuid>) onto <body> before React
          hydrates, which React reports as a server/client mismatch. The
          markup we render is identical on both sides; this silences only
          this element's attribute diff, not its children. */}
      <body
        suppressHydrationWarning
        className={`min-h-screen ${inter.className}`}
      >
        {children}
      </body>
    </html>
  );
}
