"use client"

import Link from "next/link"
import {Button} from "@/components/ui/button"
import {Car} from "lucide-react"
import {usePathname} from "next/navigation"
import Image from "next/image";
import type React from "react";

export function Navbar() {
    const pathname = usePathname()
    const isHomePage = pathname === "/"
    const isFerramentaPage = pathname === "/ferramenta"

    return (
        <nav className={`flex justify-between items-center py-4 ${isHomePage ? "" : "container mx-auto"}`}>
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
            <div className="flex gap-4 items-center">
                <Link href="/sobre" className={`hover:underline ${isHomePage ? "text-white" : "text-gray-700"}`}>
                    Sobre
                </Link>
                <Link href="/contacto" className={`hover:underline ${isHomePage ? "text-white" : "text-gray-700"}`}>
                    Contacto
                </Link>
                <Link href="/ferramenta">
                    {!isFerramentaPage ? (
                        isHomePage ? (
                            <Button variant="outline" className="bg-white/10 text-white border-white/20 hover:bg-white/20">
                                Aceder Ferramenta
                            </Button>
                        ) : (
                            <Button variant="default" className="bg-blue-600 text-white hover:bg-blue-700">
                                Aceder Ferramenta
                            </Button>
                        )
                    ) : null
                    }
                </Link>
            </div>
        </nav>
    )
}

