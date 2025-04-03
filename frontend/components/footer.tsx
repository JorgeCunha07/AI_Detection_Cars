import Link from "next/link"
import { Car } from "lucide-react"

export function Footer() {
  return (
    <footer className="bg-gray-900 text-white py-12">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center gap-2 mb-6 md:mb-0">
            <Car className="h-6 w-6" />
            <span className="text-xl font-bold">Innov8</span>
          </div>
          <div className="flex gap-8">
            <Link href="/sobre" className="hover:text-blue-400">
              Sobre
            </Link>
            <Link href="/contato" className="hover:text-blue-400">
              Contato
            </Link>
            <Link href="/privacidade" className="hover:text-blue-400">
              Privacidade
            </Link>
            <Link href="/termos" className="hover:text-blue-400">
              Termos
            </Link>
          </div>
        </div>
        <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
          <p>© {new Date().getFullYear()} Innov8. Todos os direitos reservados.</p>
        </div>
      </div>
    </footer>
  )
}

