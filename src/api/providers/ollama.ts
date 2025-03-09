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
		console.info(`createMessage(${messages.length})`)
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
						name: "list_code_definition_names",
						description:
							"Request to list definition names (classes, functions, methods, etc.) used in source code files at the top level of the specified directory. This tool provides insights into the codebase structure and important constructs, encapsulating high-level concepts and relationships that are crucial for understanding the overall architecture",
						parameters: {
							type: "object",
							properties: {
								path: { type: "string" },
							},
							required: ["path"],
							additionalProperties: false,
							$schema: "http://json-schema.org/draft-07/schema#",
						},
					},
				},
				{
					type: "function",
					function: {
						name: "list_files",
						description:
							"Request to list files and directories within the specified directory. If recursive is true, it will list all files and directories recursively. If recursive is false or not provided, it will only list the top-level contents. Do not use this tool to confirm the existence of files you may have created, as the user will let you know if the files were created successfully or not",
						parameters: {
							type: "object",
							properties: {
								path: { type: "string" },
								recursive: { type: "boolean" },
							},
							required: ["path"],
							additionalProperties: false,
							$schema: "http://json-schema.org/draft-07/schema#",
						},
					},
				},
				{
					type: "function",
					function: {
						name: "read_file",
						description: "Read a file",
						parameters: {
							type: "object",
							properties: {
								path: { type: "string" },
							},
							required: ["path"],
							additionalProperties: false,
							$schema: "http://json-schema.org/draft-07/schema#",
						},
					},
				},
				{
					type: "function",
					function: {
						name: "search_files",
						description:
							"Request to perform a regex search across files in a specified directory, providing context-rich results. This tool searches for patterns or specific content across multiple files, displaying each match with encapsulating context",
						parameters: {
							type: "object",
							properties: {
								path: { type: "string" },
								regex: { type: "string" },
								file_pattern: { type: "string" },
							},
							required: ["path", "regex"],
							additionalProperties: false,
							$schema: "http://json-schema.org/draft-07/schema#",
						},
					},
				},
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
				{
					type: "function",
					function: {
						name: "apply_diff",
						description: `Request to replace existing code using a search and replace block.
This tool allows for precise, surgical replaces to files by specifying exactly what content to search for and what to replace it with.
The tool will maintain proper indentation and formatting while making changes.
Only a single operation is allowed per tool use.
The SEARCH section must exactly match existing content including whitespace and indentation.
If you're not confident in the exact content to search for, use the read_file tool first to get the exact content.
When applying the diffs, be extra careful to remember to change any closing brackets or other syntax that may be affected by the diff farther down in the file`,
						parameters: {
							type: "object",
							properties: {
								path: { type: "string" },
								diff: { type: "string" },
								start_line: { type: "number" },
								end_line: { type: "number" },
							},
							required: ["path", "diff", "start_line", "end_line"],
							additionalProperties: false,
							$schema: "http://json-schema.org/draft-07/schema#",
						},
					},
				},
				{
					type: "function",
					function: {
						name: "execute_command",
						description:
							"Request to execute a CLI command on the system. Use this when you need to perform system operations or run specific commands to accomplish any step in the user's task. You must tailor your command to the user's system and provide a clear explanation of what the command does. For command chaining, use the appropriate chaining syntax for the user's shell. Prefer to execute complex CLI commands over creating executable scripts, as they are more flexible and easier to run",
						parameters: {
							type: "object",
							properties: {
								command: { type: "string" },
							},
							required: ["command"],
							additionalProperties: false,
							$schema: "http://json-schema.org/draft-07/schema#",
						},
					},
				},
				{
					type: "function",
					function: {
						name: "ask_followup_question",
						description:
							"Ask the user a question to gather additional information needed to complete the task. This tool should be used when you encounter ambiguities, need clarification, or require more details to proceed effectively. It allows for interactive problem-solving by enabling direct communication with the user. Use this tool judiciously to maintain a balance between gathering necessary information and avoiding excessive back-and-forth",
						parameters: {
							type: "object",
							properties: {
								question: { type: "string" },
							},
							required: ["question"],
							additionalProperties: false,
							$schema: "http://json-schema.org/draft-07/schema#",
						},
					},
				},
				{
					type: "function",
					function: {
						name: "attempt_completion",
						description: `After each tool use, the user will respond with the result of that tool use, i.e. if it succeeded or failed, along with any reasons for failure. Once you've received the results of tool uses and can confirm that the task is complete, use this tool to present the result of your work to the user. Optionally you may provide a CLI command to showcase the result of your work. The user may respond with feedback if they are not satisfied with the result, which you can use to make improvements and try again.
IMPORTANT NOTE: This tool CANNOT be used until you've confirmed from the user that any previous tool uses were successful. Failure to do so will result in code corruption and system failure. Before using this tool, you must ask yourself in <thinking></thinking> tags if you've confirmed from the user that any previous tool uses were successful. If not, then DO NOT use this tool
Parameters:

- result: (required) The result of the task. Formulate this result in a way that is final and does not require further input from the user. Don't end your result with questions or offers for further assistance.
- command: (optional) A CLI command to execute to show a live demo of the result to the user. For example, use \`open index.html\` to display a created html website, or \`open localhost:3000\` to display a locally running development server. But DO NOT use commands like \`echo\` or \`cat\` that merely print text. This command should be valid for the current operating system. Ensure the command is properly formatted and does not contain any harmful instructions`,
						parameters: {
							type: "object",
							properties: {
								result: { type: "string" },
								command: { type: "string" },
							},
							required: ["result"],
							additionalProperties: false,
							$schema: "http://json-schema.org/draft-07/schema#",
						},
					},
				},
				{
					type: "function",
					function: {
						name: "switch_mode",
						description: `Request to switch to a different mode. This tool allows modes to request switching to another mode when needed, such as switching to Code mode to make code changes. The user must approve the mode switch`,
						parameters: {
							type: "object",
							properties: {
								mode_slug: { type: "string" },
								reason: { type: "string" },
							},
							required: ["mode_slug"],
							additionalProperties: false,
							$schema: "http://json-schema.org/draft-07/schema#",
						},
					},
				},
				{
					type: "function",
					function: {
						name: "new_task",
						description: `Create a new task with a specified starting mode and initial message. This tool instructs the system to create a new Cline instance in the given mode with the provided message`,
						parameters: {
							type: "object",
							properties: {
								mode: { type: "string" },
								message: { type: "string" },
							},
							required: ["mode", "message"],
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
				console.info(`MESSAGE: ${choice.message.content}`)
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
					console.info(`TOOL_CALL: ${tool.function}`)
					const args = JSON.parse(tool.function.arguments) as Record<string, unknown>
					const xml = `<${tool.function.name}>
						${convertToXML(args)}
					</${tool.function.name}>`
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
