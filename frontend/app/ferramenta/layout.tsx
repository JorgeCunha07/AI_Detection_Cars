import type React from "react"
import { Footer } from "@/components/footer"
import { Navbar } from "@/components/navbar"

export default function FerramentaLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen flex flex-col">
      <div className="container mx-auto px-4">
        <Navbar />
      </div>
      <main className="flex-grow">{children}</main>
      <Footer />
    </div>
  )
}

