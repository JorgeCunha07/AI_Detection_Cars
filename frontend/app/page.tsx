import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowRight, Car, Brain, Award, School } from "lucide-react"
import Image from "next/image"
import { Footer } from "@/components/footer"
import type React from "react";

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <header className="bg-gradient-to-r from-blue-600 via-blue-500 to-purple-600 text-white">
        <div className="container mx-auto px-4 pb-16 md:pb-24">
          <nav className="flex justify-between items-center py-4 px-8">
            <div className="flex items-center gap-2">
              <Link href="/">
                  <Image
                      src="/innov8.png"
                      alt="Aluno a aprender com IA"
                      width={100}
                      height={50}
                      className="object-cover"
                  />
              </Link>
            </div>
            <div className="flex gap-4">
              <Link href="/sobre" className="hover:underline">
                Sobre
              </Link>
              <Link href="/contacto" className="hover:underline">
                Contacto
              </Link>
            </div>
          </nav>

          <div className="flex flex-col md:flex-row items-center gap-12 pt-24">
            <div className="md:w-1/2 space-y-6">
              <h1 className="text-4xl md:text-6xl font-bold leading-tight">
                Revolucionar o Ensino de Condução com Inteligência Artificial
              </h1>
              <p className="text-xl opacity-90">
                Ajudamos os alunos a prepararem-se melhor para os exames de condução através da análise de imagens e
                feedback inteligente.
              </p>
              <div className="flex gap-4 pt-4">
                <Link href="/ferramenta">
                  <Button className="bg-white text-blue-600 hover:bg-blue-50">
                    Experimentar Ferramenta <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
                <Link href="/sobre">
                  <Button variant="outline" className="border-white text-white hover:bg-white/50">
                    Saiba Mais
                  </Button>
                </Link>
              </div>
            </div>
            <div className="md:w-1/2">
              <div className="relative w-full h-[300px] md:h-[400px] rounded-lg overflow-hidden shadow-2xl">
                <Image
                  src="/homepage.png"
                  alt="Aluno a aprender com IA"
                  fill
                  className="object-cover"
                />
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Features Section */}
      <section className="py-16 md:py-24 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Como a Innov8 Transforma a Aprendizagem</h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              A nossa tecnologia de ponta utiliza inteligência artificial para identificar e corrigir erros comuns,
              preparando melhor os alunos para os exames de condução.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white p-8 rounded-xl shadow-md">
              <div className="bg-blue-100 p-3 rounded-full w-fit mb-6">
                <Brain className="h-8 w-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-bold mb-3">Análise Inteligente de Imagens</h3>
              <p className="text-gray-600">
                A nossa IA analisa imagens de situações de trânsito, identificando potenciais perigos e decisões
                corretas que o condutor deve tomar.
              </p>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-md">
              <div className="bg-purple-100 p-3 rounded-full w-fit mb-6">
                <School className="h-8 w-8 text-purple-600" />
              </div>
              <h3 className="text-xl font-bold mb-3">Feedback Personalizado</h3>
              <p className="text-gray-600">
                Geramos explicações detalhadas e personalizadas para cada situação, ajudando os alunos a compreenderem
                melhor as regras e práticas de condução segura.
              </p>
            </div>

            <div className="bg-white p-8 rounded-xl shadow-md">
              <div className="bg-green-100 p-3 rounded-full w-fit mb-6">
                <Award className="h-8 w-8 text-green-600" />
              </div>
              <h3 className="text-xl font-bold mb-3">Preparação para o Exame</h3>
              <p className="text-gray-600">
                Simulamos situações reais de exame, preparando os alunos para tomarem decisões rápidas e corretas
                durante a avaliação prática.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 md:py-24 bg-white">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <div>
              <p className="text-5xl font-bold text-blue-600 mb-2">85%</p>
              <p className="text-xl text-gray-600">Aumento na taxa de aprovação</p>
            </div>
            <div>
              <p className="text-5xl font-bold text-purple-600 mb-2">10.000+</p>
              <p className="text-xl text-gray-600">Alunos beneficiados</p>
            </div>
            <div>
              <p className="text-5xl font-bold text-green-600 mb-2">98%</p>
              <p className="text-xl text-gray-600">Satisfação dos instrutores</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 md:py-24 bg-gradient-to-r from-blue-600 to-purple-600 text-white">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">Pronto para revolucionar o ensino de condução?</h2>
          <p className="text-xl mb-8 max-w-3xl mx-auto">
            Junte-se às escolas de condução que já estão a utilizar a nossa tecnologia para preparar melhor os seus
            alunos.
          </p>
          <Link href="/ferramenta">
            <Button className="bg-white text-blue-600 hover:bg-blue-50 text-lg px-8 py-6">
              Experimentar a Nossa Ferramenta
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <Footer />
    </div>
  )
}