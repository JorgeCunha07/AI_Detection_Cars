import { type NextRequest, NextResponse } from "next/server"

// URL base da API
const API_BASE_URL = "http://localhost:8000" // Ajuste para o endereço correto da sua API

export async function GET(request: NextRequest, { params }: { params: { path: string[] } }) {
  const path = params.path.join("/")
  const url = `${API_BASE_URL}/${path}`

  try {
    const response = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
      },
    })

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error(`Erro ao fazer proxy para ${url}:`, error)
    return NextResponse.json({ error: "Erro ao comunicar com o servidor" }, { status: 500 })
  }
}

export async function POST(request: NextRequest, { params }: { params: { path: string[] } }) {
  const path = params.path.join("/")
  const url = `${API_BASE_URL}/${path}`
  const body = await request.json()

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    })

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error(`Erro ao fazer proxy para ${url}:`, error)
    return NextResponse.json({ error: "Erro ao comunicar com o servidor" }, { status: 500 })
  }
}

