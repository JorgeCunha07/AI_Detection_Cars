import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

const labelsForDescription = [
  "carro",
  "autocarro",
  "camião",
  "peão",
  "ciclista",
  "passadeira",
  "semáforo",
  "sinal de stop",
  "sinal de limite de velocidade",
  "sinal de passadeira",
  "céu limpo",
  "nublado",
  "chuva",
  "nevoeiro",
  "vento",
  "neve",
  "amanhecer",
  "anoitecer",
  "dia",
  "noite",
  "zona residencial",
  "parque de estacionamento",
  "túnel",
  "cidade",
  "autoestrada",
];

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function translateLabel(label: string): string {
  switch (label.toLowerCase()) {
    case "city street":
      return "cidade";
    case "night":
      return "noite";
    case "area/drivable":
      return "faixa de rodagem";
    case "clear":
      return "céu limpo";
    case "traffic light":
      return "semáforo";
    case "area/alternative":
      return "via alternativa";
    case "bike":
      return "ciclista";
    case "lane/crosswalk":
      return "passadeira";
    case "lane/double other":
      return "via delimitada por outro tipo de linhas duplas";
    case "lane/double white":
      return "via delimitada por linhas duplas brancas";
    case "lane/double yellow":
      return "via delimitada por linhas duplas amarelas";
    case "lane/road curb":
      return "separador de via";
    case "lane/single other":
      return "via delimitada por outro tipo de linhas";
    case "lane/single white":
      return "via delimitada por linhas brancas";
    case "lane/single yellow":
      return "via delimitada por linhas amarelas";
    case "motor":
      return "mota";
    case "rider":
      return "ciclista";
    case "train":
      return "comboio";
    case "highway":
      return "autoestrada";
    case "daytime":
      return "dia";
    case "pedestrian":
      return "peão";
    case "car":
      return "carro";
    case "truck":
      return "camião";
    case "bus":
      return "autocarro";
    case "bycicle":
      return "ciclista";
    case "crosswalk":
      return "passadeira";
    case "crosswalk signal":
      return "sinal de passadeira";
    case "speedlimit signal":
      return "sinal de limite de velocidade";
    case "stop signal":
      return "sinal de stop";
    case "trafic light signal":
      return "semáforo";
    case "overcast":
      return "nublado";
    default:
      return label;
  }
}

export function getLabelPlural(label: string): string {
  switch (label.toLowerCase()) {
    case "peão":
      return "peões";
    case "carro":
      return "carros";
    case "camião":
      return "camiões";
    case "autocarro":
      return "autocarros";
    case "ciclista":
      return "ciclistas";
    case "passadeira":
      return "passadeiras";
    case "semáforo":
      return "semáforos";
    default:
      return label;
  }
}

export function transformLabelsWithSpaces(labels: string[]): string[] {
  return labels.map((label) => label.replace(/ /g, "_"));
}

export function isLabelValidForDescription(label: string): boolean {
  return labelsForDescription.includes(label);
}
