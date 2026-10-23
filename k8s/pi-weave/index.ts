/**
 * Trace pi runs to W&B Weave.
 *
 *   invoke_agent   one per pi run (a kaggler iteration), carries the prompt and final answer
 *   chat <model>   one per pi turn, carries the assistant output and token usage
 *   execute_tool   one per tool call, nested under the turn that issued it
 *
 * Requires WANDB_API_KEY, WANDB_ENTITY and WANDB_PROJECT. KAGGLER_NAME names the agent.
 */

import type { AssistantMessage, ImageContent, TextContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import * as weave from "weave";
import type { Conversation, LLM, MessagePart, Tool, Turn } from "weave";

const MAX_RESULT_CHARS = 20_000;

function assistantParts(content: AssistantMessage["content"]): MessagePart[] {
	const parts: MessagePart[] = [];
	for (const block of content) {
		if (block.type === "text" && block.text) parts.push({ type: "text", content: block.text });
		if (block.type === "thinking" && block.thinking) parts.push({ type: "reasoning", content: block.thinking });
		if (block.type === "toolCall")
			parts.push({ type: "tool_call", toolCallId: block.id, toolName: block.name, arguments: JSON.stringify(block.arguments) });
	}
	return parts;
}

function textOf(content: (TextContent | ImageContent)[] | AssistantMessage["content"]): string {
	return content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("\n");
}

function resultText(result: { content: (TextContent | ImageContent)[] }): string {
	const text = textOf(result.content);
	return text.length > MAX_RESULT_CHARS ? `${text.slice(0, MAX_RESULT_CHARS)}\n[truncated]` : text;
}

export default function (pi: ExtensionAPI) {
	let conversation: Conversation | undefined;
	let run: Turn | undefined;
	let llm: LLM | undefined;
	let lastAnswer = "";
	const tools = new Map<string, Tool>();

	pi.on("session_start", async (_event, ctx) => {
		// weave.init prints the project URL with console.log, but JSON mode reserves stdout for events.
		const log = console.log;
		console.log = console.error;
		await weave.init(`${process.env.WANDB_ENTITY}/${process.env.WANDB_PROJECT}`);
		console.log = log;
		conversation = weave.runIsolated(() =>
			weave.startConversation({
				agentName: process.env.KAGGLER_NAME ?? "pi",
				conversationId: ctx.sessionManager.getSessionId(),
			}),
		);
	});

	pi.on("before_agent_start", (event, ctx) => {
		lastAnswer = "";
		run = conversation!.startTurn({
			model: ctx.model?.id,
			userMessage: event.messageType === "user" ? event.prompt : undefined,
			systemInstructions: [event.systemPrompt],
		});
	});

	pi.on("turn_start", (_event, ctx) => {
		llm = run!.startLLM({ model: ctx.model?.id ?? "", providerName: ctx.model?.provider });
	});

	pi.on("message_end", (event) => {
		if (event.message.role !== "assistant") return;
		const message = event.message as AssistantMessage;
		const usage = message.usage;
		llm!.record({
			outputMessages: [{ role: "assistant", parts: assistantParts(message.content) }],
			responseModel: message.model,
			finishReasons: [message.stopReason],
			usage: {
				inputTokens: usage.input + usage.cacheRead + usage.cacheWrite,
				outputTokens: usage.output,
				reasoningTokens: usage.reasoning ?? 0,
				cacheReadInputTokens: usage.cacheRead,
				cacheCreationInputTokens: usage.cacheWrite,
				totalTokens: usage.totalTokens,
			},
		});
		llm!.setAttributes({ "pi.usage.cost_usd": usage.cost.total });
		const text = textOf(message.content);
		if (text) lastAnswer = text;
	});

	pi.on("tool_execution_start", (event) => {
		tools.set(event.toolCallId, llm!.startTool({ name: event.toolName, toolCallId: event.toolCallId, args: event.args }));
	});

	pi.on("tool_execution_end", (event) => {
		const text = resultText(event.result);
		tools.get(event.toolCallId)!.end(event.isError ? { error: new Error(text) } : { result: text });
		tools.delete(event.toolCallId);
	});

	pi.on("turn_end", () => {
		llm?.end();
		llm = undefined;
	});

	pi.on("agent_settled", async (event) => {
		llm?.end();
		llm = undefined;
		run!.record({ outputMessages: [{ role: "assistant", content: lastAnswer }] });
		run!.end(event.aborted ? { error: new Error("run aborted") } : undefined);
		run = undefined;
		await weave.flushOTel();
	});

	pi.on("session_shutdown", async () => {
		run?.end({ error: new Error("session shut down mid-run") });
		run = undefined;
		conversation?.end();
		await weave.flushOTel();
	});
}
