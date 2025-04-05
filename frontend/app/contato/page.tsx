import { Navbar } from "@/components/navbar"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Mail, Phone, MapPin, Send } from "lucide-react"
import { Car } from "lucide-react"
import Image from "next/image";

export default function ContatoPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <div className="container mx-auto px-4">
        <Navbar />
      </div>

      <main className="flex-grow">
        {/* Hero Section */}
        <section className="bg-blue-50 py-16">
          <div className="container mx-auto px-4">
            <div className="max-w-3xl mx-auto text-center">
              <h1 className="text-4xl md:text-5xl font-bold mb-6 text-blue-600">Entre em Contato</h1>
              <p className="text-xl text-gray-700">
                 Estamos aqui para responder às suas dúvidas e ajudar a sua escola de condução a implementar a nossa tecnologia.
              </p>
            </div>
          </div>
        </section>

        {/* Contact Form Section */}
        <section className="py-16">
          <div className="container mx-auto px-4">
            <div className="flex flex-col md:flex-row gap-12">
              <div className="md:w-1/2">
                <h2 className="text-3xl font-bold mb-6 text-blue-600">Envie uma Mensagem</h2>
                <Card>
                  <CardContent className="pt-6">
                    <form className="space-y-6">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-2">
                          <label htmlFor="nome" className="text-sm font-medium">
                            Nome
                          </label>
                          <input
                            id="nome"
                            type="text"
                            className="w-full p-3 border rounded-md"
                            placeholder="Seu nome"
                          />
                        </div>
                        <div className="space-y-2">
                          <label htmlFor="email" className="text-sm font-medium">
                            Email
                          </label>
                          <input
                            id="email"
                            type="email"
                            className="w-full p-3 border rounded-md"
                            placeholder="seu@email.com"
                          />
                        </div>
                      </div>
                      <div className="space-y-2">
                        <label htmlFor="assunto" className="text-sm font-medium">
                          Assunto
                        </label>
                        <input
                          id="assunto"
                          type="text"
                          className="w-full p-3 border rounded-md"
                          placeholder="Assunto da mensagem"
                        />
                      </div>
                      <div className="space-y-2">
                        <label htmlFor="mensagem" className="text-sm font-medium">
                          Mensagem
                        </label>
                        <Textarea id="mensagem" placeholder="Sua mensagem" className="min-h-32" />
                      </div>
                      <Button className="w-full bg-blue-600 hover:bg-blue-700">
                        Enviar Mensagem <Send className="ml-2 h-4 w-4" />
                      </Button>
                    </form>
                  </CardContent>
                </Card>
              </div>
              <div className="md:w-1/2">
                <h2 className="text-3xl font-bold mb-6 text-blue-600">Informações de Contato</h2>
                <div className="space-y-8">
                  <div className="flex items-start">
                    <div className="bg-blue-100 p-3 rounded-full mr-4">
                      <Mail className="h-6 w-6 text-blue-600" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold mb-2">Email</h3>
                      <p className="text-gray-700">innov8@email.com</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="bg-blue-100 p-3 rounded-full mr-4">
                      <Phone className="h-6 w-6 text-blue-600" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold mb-2">Telefone</h3>
                      <p className="text-gray-700">(+351) XXX XXX XXX</p>
                    </div>
                  </div>
                  <div className="flex items-start">
                    <div className="bg-blue-100 p-3 rounded-full mr-4">
                      <MapPin className="h-6 w-6 text-blue-600" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold mb-2">Endereço</h3>
                      <p className="text-gray-700">
                        R. Dr. António Bernardino de Almeida 431
                        <br />
                        Porto
                        <br />
                        Portugal
                      </p>
                    </div>
                  </div>
                </div>
                <div className="mt-8 h-64 bg-gray-200 lg">
                  <div className="relative w-full h-[250px] md:h-[300px] rounded-lg overflow-hidden shadow-xl">
                   <Image
                  src="/isepMapa.png"
                  alt="Mapa"
                  fill
                  className="object-cover"
                  />
                    </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center gap-2 mb-6 md:mb-0">
              <Car className="h-6 w-6" />
              <span className="text-xl font-bold">Innov8</span>
            </div>
            <div className="flex gap-8">
              <a href="/sobre" className="hover:text-blue-400">
                Sobre
              </a>
              <a href="/contato" className="hover:text-blue-400">
                Contato
              </a>
              <a href="/privacidade" className="hover:text-blue-400">
                Privacidade
              </a>
              <a href="/termos" className="hover:text-blue-400">
                Termos
              </a>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>© {new Date().getFullYear()} Innov8. Todos os direitos reservados.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

