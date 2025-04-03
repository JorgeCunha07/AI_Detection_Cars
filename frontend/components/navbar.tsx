"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Car } from "lucide-react"
import { usePathname } from "next/navigation"

export function Navbar() {
  const pathname = usePathname()
  const isHomePage = pathname === "/"

  return (
    <nav className={`flex justify-between items-center py-4 ${isHomePage ? "" : "container mx-auto"}`}>
      <div className="flex items-center gap-2">
        <Car className={`h-6 w-6 ${isHomePage ? "text-white" : "text-blue-600"}`} />
        <span className={`text-xl font-bold ${isHomePage ? "text-white" : "text-blue-600"}`}>Innov8</span>
      </div>
      <div className="flex gap-4 items-center">
        <Link href="/sobre" className={`hover:underline ${isHomePage ? "text-white" : "text-gray-700"}`}>
          Sobre
        </Link>
        <Link href="/contato" className={`hover:underline ${isHomePage ? "text-white" : "text-gray-700"}`}>
          Contato
        </Link>
        <Link href="/ferramenta">
          {isHomePage ? (
            <Button variant="outline" className="bg-white/10 text-white border-white/20 hover:bg-white/20">
              Acessar Ferramenta
            </Button>
          ) : (
            <Button variant="default" className="bg-blue-600 text-white hover:bg-blue-700">
              Acessar Ferramenta
            </Button>
          )}
        </Link>
      </div>
    </nav>
  )
}

