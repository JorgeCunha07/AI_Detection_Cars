"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Upload, ImageIcon, Loader2 } from "lucide-react"
import Image from "next/image"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface ImageUploadProps {
  onImageAnalyzed: (description: string) => void
}

export function ImageUpload({ onImageAnalyzed }: ImageUploadProps) {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [selectedModel, setSelectedModel] = useState<string | undefined>()
  const [imageFile, setImageFile] = useState<File | null>(null)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const objectUrl = URL.createObjectURL(file)
    setImageUrl(objectUrl)
    setImageFile(file)
  }

  const handleAnalyzeImage = async () => {
    if (!imageFile) return

    setIsAnalyzing(true)

    try {
      // Em um ambiente real, usaríamos o serviço de API
      // Aqui estamos simulando uma resposta após 2 segundos
      setTimeout(() => {
        const simulatedDescription =
          "This image shows a stunning landscape with mountains in the background and a lake in the foreground. The sky has beautiful cloud formations, and the colors are vibrant with blues, greens, and hints of orange from the setting sun. The composition creates a sense of depth and tranquility, making it an excellent example of nature photography."

        onImageAnalyzed(simulatedDescription)
        setIsAnalyzing(false)
      }, 2000)
    } catch (error) {
      console.error("Error analyzing image:", error)
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-center">
        <label
          htmlFor="image-upload"
          className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 dark:hover:bg-gray-800 dark:bg-gray-900 border-gray-300 dark:border-gray-700 transition-all"
        >
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <Upload className="w-10 h-10 mb-3 text-gray-400" />
            <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
              <span className="font-semibold">Click para fazer upload</span> ou arraste e solte
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">PNG ou JPG (MAX. 10MB)</p>
          </div>
          <input id="image-upload" type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
        </label>
      </div>

      {imageUrl && (
        <div className="flex flex-col items-center space-y-4">
          <div className="flex flex-wrap justify-center w-full gap-4">
            <Image
              src={imageUrl || "/placeholder.svg"}
              alt="Imagem carregada"
              width={350}
              height={350}
              className="object-contain rounded-lg shadow-md"
            />
            <Image
              src={imageUrl || "/placeholder.svg"}
              alt="Imagem carregada com processamento"
              width={350}
              height={350}
              className="object-contain rounded-lg shadow-md"
            />
          </div>
          <div className="flex gap-2.5">
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Selecione o modelo" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectLabel>Modelo</SelectLabel>
                  <SelectItem value="nosso">Nosso</SelectItem>
                  <SelectItem value="prefeito">Pré-feito</SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>
            <Button
              onClick={handleAnalyzeImage}
              disabled={isAnalyzing || !imageUrl}
              className="flex items-center space-x-2 bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500 text-white"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Analisando...</span>
                </>
              ) : (
                <>
                  <ImageIcon className="w-4 h-4" />
                  <span>Analisar Imagem</span>
                </>
              )}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

