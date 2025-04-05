import { Navbar } from "@/components/navbar"
import { Button } from "@/components/ui/button"
import { ArrowRight, CheckCircle } from "lucide-react"
import Image from "next/image"
import Link from "next/link"
import { Car } from "lucide-react"

export default function SobrePage() {
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
              <h1 className="text-4xl md:text-5xl font-bold mb-6 text-blue-600">Sobre a Innov8</h1>
              <p className="text-xl text-gray-700 mb-8">
                Transformando o ensino de condução com tecnologia de ponta e inteligência artificial.
              </p>
            </div>
          </div>
        </section>

        {/* Mission Section */}
        <section className="py-16">
          <div className="container mx-auto px-4">
            <div className="flex flex-col md:flex-row items-center gap-12">
              <div className="md:w-1/2">
                <h2 className="text-3xl font-bold mb-6 text-blue-600">Nossa Missão</h2>
                <p className="text-lg text-gray-700 mb-6">
                  Na Innov8, acreditamos que a tecnologia pode revolucionar a forma como as pessoas aprendem a dirigir.
                  Nossa missão é tornar as estradas mais seguras, preparando melhor os futuros condutores através de
                  análise inteligente e feedback personalizado.
                </p>
                <p className="text-lg text-gray-700 mb-6">
                  A Innov8 foi fundada em 2024 por um grupo de estudantes de engenharia de inteligência artificial (IA).
                  A empresa nasceu da perceção de que muitos alunos têm dificuldade em compreender situações complexas
                  de trânsito e em tomar decisões rápidas durante os exames.
                </p>
                <p className="text-lg text-gray-700">
                  Hoje, nossa tecnologia é utilizada por mais de 200 escolas de condução em todo o país, ajudando
                  milhares de alunos a se tornarem condutores mais seguros e confiantes.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Technology Section */}
        <section className="py-16 bg-gray-50">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold mb-12 text-center text-blue-600">Nossa Tecnologia</h2>
            <div className="grid md:grid-cols-2 gap-12">
              <div className="bg-white p-8 rounded-xl shadow-md">
                <h3 className="text-2xl font-bold mb-4 text-blue-600">Análise de Imagem</h3>
                <p className="text-gray-700 mb-6">
                  Nossos modelos de IA são treinados com milhares de imagens de situações reais de trânsito, permitindo
                  identificar elementos cruciais como sinais, faixas, outros veículos e pedestres.
                </p>
                <ul className="space-y-3">
                  {[
                    "Detecção de sinalização",
                    "Identificação de perigos",
                    "Análise de posicionamento",
                    "Verificação de distâncias",
                  ].map((item, index) => (
                    <li key={index} className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="bg-white p-8 rounded-xl shadow-md">
                <h3 className="text-2xl font-bold mb-4 text-purple-600">Geração de Texto</h3>
                <p className="text-gray-700 mb-6">
                  Os nossos modelos de linguagem avançados permitem-nos
                  fornecer explicações claras e didáticas sobre cada situação de trânsito,
                  adaptando o conteúdo ao nível de conhecimento do aluno.
                </p>
                <ul className="space-y-3">
                  {[
                    "Feedback personalizado",
                    "Explicações detalhadas",
                    "Dicas práticas",
                    "Correção de erros comuns",
                  ].map((item, index) => (
                    <li key={index} className="flex items-start">
                      <CheckCircle className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-700">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Team Section */}
        <section className="py-16">
          <div className="container mx-auto px-4">
            <h2 className="text-3xl font-bold mb-12 text-center text-blue-600">Nossa Equipa</h2>
            <div className="grid md:grid-cols-3 gap-8">
              {[
                {
                  name: "Jorge Cunha",
                  role: "Full Stack Software Developer",
                },
                {
                  name: "Miguel Ramos",
                  role: "Software Developer",
                },
                {
                  name: "Carlos Moutinho",
                  role: "RPA Developer",
                },
                {
                  name: "Andre Sousa",
                  role: "Software Developer",
                },
                {
                  name: "Sabrina Pacheco",
                  role: "Software Developer",
                }
              ].map((member, index) => (
                <div key={index} className="bg-white p-6 rounded-xl shadow-md text-center">
                  <div className="w-32 h-32 mx-auto mb-4 rounded-full overflow-hidden">
                    <Image
                      src={`/placeholder.svg?height=128&width=128`}
                      alt={member.name}
                      width={128}
                      height={128}
                      className="object-cover"
                    />
                  </div>
                  <h3 className="text-xl font-bold mb-1">{member.name}</h3>
                  <p className="text-blue-600 mb-3">{member.role}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-16 bg-blue-600 text-white">
          <div className="container mx-auto px-4 text-center">
            <h2 className="text-3xl font-bold mb-6">Pronto para transformar o aprendizado de condução?</h2>
            <p className="text-xl mb-8 max-w-3xl mx-auto">
              Experimente nossa ferramenta de análise de imagens e veja como podemos ajudar seus alunos a se prepararem
              melhor para o exame.
            </p>
            <Link href="/ferramenta">
              <Button className="bg-white text-blue-600 hover:bg-blue-50 text-lg px-8 py-6">
                Experimentar Agora <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
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
    </div>
  )
}

