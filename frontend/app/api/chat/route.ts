import { NextResponse } from "next/server"
import { useEffect, useState } from 'react'

export async function POST(request: Request) {
  try {
    const { message, history } = await request.json()

    const simulatedResponses = [
      "This appears to be a beautiful landscape photo...",
      "I notice there are several interesting elements...",
      // ...other responses...
    ]

    // Use a deterministic way to select response based on input
    const responseIndex = Math.abs(message.length) % simulatedResponses.length

    await new Promise((resolve) => setTimeout(resolve, 500))

    return NextResponse.json({ response: simulatedResponses[responseIndex] })
  } catch (error) {
    console.error("Erro:", error)
    return NextResponse.json({ error: "Erro ao processar mensagem" }, { status: 500 })
  }
}

export function MyComponent() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return null // or a loading state
  }

  // Your component logic here
  return (
    // ...
  )
}

