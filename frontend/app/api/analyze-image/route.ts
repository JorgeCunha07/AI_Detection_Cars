import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const imageFile = formData.get("image") as File

    if (!imageFile) {
      return NextResponse.json({ error: "No image file provided" }, { status: 400 })
    }

    // Converte o arquivo para buffer
    const bytes = await imageFile.arrayBuffer()
    const buffer = Buffer.from(bytes)

    // Em um ambiente real, usaríamos o OpenAI para análise
    // Aqui estamos simulando uma resposta
    const description =
      "This image shows a stunning landscape with mountains in the background and a lake in the foreground. The sky has beautiful cloud formations, and the colors are vibrant with blues, greens, and hints of orange from the setting sun."

    return NextResponse.json({ description })
  } catch (error) {
    console.error("Error analyzing image:", error)
    return NextResponse.json({ error: "Failed to analyze image" }, { status: 500 })
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
}

