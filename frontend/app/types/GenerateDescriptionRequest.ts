export default interface GenerateDescriptionRequest {
    labels: string[],
    mode: string,
    topk?: number,
    beam_width?: number,
    temperature?: number
}