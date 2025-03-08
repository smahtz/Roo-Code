import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI from "openai"
import axios from "axios"

import { SingleCompletionHandler } from "../"
import { ApiHandlerOptions, ModelInfo, openAiModelInfoSaneDefaults } from "../../shared/api"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { convertToR1Format } from "../transform/r1-format"
import { ApiStream } from "../transform/stream"
import { DEEP_SEEK_DEFAULT_TEMPERATURE } from "./constants"
import { XmlMatcher } from "../../utils/xml-matcher"
import { BaseProvider } from "./base-provider"

export class OllamaHandler extends BaseProvider implements SingleCompletionHandler {
	protected options: ApiHandlerOptions
	private client: OpenAI

	constructor(options: ApiHandlerOptions) {
		super()
		this.options = options
		this.client = new OpenAI({
			baseURL: (this.options.ollamaBaseUrl || "http://localhost:11434") + "/v1",
			apiKey: "ollama",
		})
	}

	override async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const modelId = this.getModel().id
		const useR1Format = modelId.toLowerCase().includes("deepseek-r1")

		// I've seen better results when user request comes at the end of user content
		// User messages here seem to have the user's request/task first, then context data
		// I was seeing several local models reply like there was nothing to do, but
		// flipping content seems to focus on the <task>...</task> part at the end of user message
		const flippedMessages = messages.map((m) => {
			if (m.role === "user" && Array.isArray(m.content)) {
				m.content = m.content.reverse()
			}
			return m
		})

		const openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "system", content: systemPrompt },
			...(useR1Format ? convertToR1Format(flippedMessages) : convertToOpenAiMessages(flippedMessages)),
		]

		const stream = await this.client.chat.completions.create({
			model: this.getModel().id,
			messages: openAiMessages,
			temperature: this.options.modelTemperature ?? 0,
			tools: [
				{
					type: "function",
					function: {
						name: "write_to_file",
						description: "Create a file",
						parameters: {
							type: "object",
							properties: {
								path: { type: "string" },
								content: { type: "string" },
								line_count: { type: "number" },
							},
							required: ["path", "content", "line_count"],
							additionalProperties: false,
							$schema: "http://json-schema.org/draft-07/schema#",
						},
					},
				},
			],
			stream: false,
		})
		const matcher = new XmlMatcher(
			"think",
			(chunk) =>
				({
					type: chunk.matched ? "reasoning" : "text",
					text: chunk.data,
				}) as const,
		)
		for (const choice of stream.choices) {
			if (choice.message.content) {
				for (const chunk of matcher.update(choice.message.content)) {
					yield chunk
				}
			} else if (choice.message.tool_calls) {
				function convertToXML(obj: any): string {
					let xml = ""

					for (const key in obj) {
						if (obj.hasOwnProperty(key)) {
							const value = obj[key]
							// If the value is an object, recursively call the function
							if (typeof value === "object" && value !== null && !Array.isArray(value)) {
								xml += `<${key}>${convertToXML(value)}</${key}>`
							} else if (Array.isArray(value)) {
								// If the value is an array, iterate over each element
								value.forEach((item: any) => {
									xml += `<${key}>${convertToXML(item)}</${key}>`
								})
							} else {
								// For primitive values, just add them to the XML string
								xml += `<${key}>${value}</${key}>`
							}
						}
					}

					return xml
				}
				for (const tool of choice.message.tool_calls) {
					const args = JSON.parse(tool.function.arguments) as Record<string, unknown>
					const xml = `<${tool.function.name}>
						${convertToXML(args)}
					</${tool.function.name}>`
					console.info("TOOL CALL", xml)
					for (const chunk of matcher.update(xml)) {
						yield chunk
					}
				}
			}
			for (const chunk of matcher.final()) {
				yield chunk
			}
		}
	}

	override getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.ollamaModelId || "",
			info: openAiModelInfoSaneDefaults,
		}
	}

	async completePrompt(prompt: string): Promise<string> {
		try {
			const modelId = this.getModel().id
			const useR1Format = modelId.toLowerCase().includes("deepseek-r1")
			const response = await this.client.chat.completions.create({
				model: this.getModel().id,
				messages: useR1Format
					? convertToR1Format([{ role: "user", content: prompt }])
					: [{ role: "user", content: prompt }],
				temperature: this.options.modelTemperature ?? (useR1Format ? DEEP_SEEK_DEFAULT_TEMPERATURE : 0),
				stream: false,
			})
			return response.choices[0]?.message.content || ""
		} catch (error) {
			if (error instanceof Error) {
				throw new Error(`Ollama completion error: ${error.message}`)
			}
			throw error
		}
	}
}

export async function getOllamaModels(baseUrl = "http://localhost:11434") {
	try {
		if (!URL.canParse(baseUrl)) {
			return []
		}

		const response = await axios.get(`${baseUrl}/api/tags`)
		const modelsArray = response.data?.models?.map((model: any) => model.name) || []
		return [...new Set<string>(modelsArray)]
	} catch (error) {
		return []
	}
}
