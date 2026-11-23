import { z } from "zod";
const region = z
  .object({
    present: z.boolean(),
    volume_cm3: z.number(),
    n_components: z.number(),
  })
  .passthrough();
const report = z
  .object({
    device: z.string(),
    detection: z.object({
      tumor_detected: z.boolean(),
      max_probability: z.number(),
      threshold: z.number(),
      n_slices_checked: z.number(),
    }),
    localization: z.object({ WT: region, TC: region, ET: region }),
    summary: z.string(),
    seconds: z.number(),
  })
  .passthrough();
const analysisBase = z.object({
  id: z.string().uuid(),
  source: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
  metadata: z.object({
    shape: z.tuple([z.number(), z.number(), z.number()]),
    orientation: z.array(z.string()),
    has_reference: z.boolean(),
  }),
});
export const analysisSchema = z.discriminatedUnion("status", [
  analysisBase.extend({
    status: z.literal("queued"),
    report: z.null(),
    error: z.null(),
  }),
  analysisBase.extend({
    status: z.literal("running"),
    report: z.null(),
    error: z.null(),
  }),
  analysisBase.extend({
    status: z.literal("completed"),
    report,
    error: z.null(),
  }),
  analysisBase.extend({
    status: z.literal("failed"),
    report: z.null(),
    error: z.string(),
  }),
]);
export type Analysis = z.infer<typeof analysisSchema>;
export const healthSchema = z.object({
  models: z.object({
    status: z.enum(["loading", "ready", "error"]),
    device: z.string().nullable(),
    error: z.string().nullable(),
  }),
  research_only: z.boolean(),
});
export const samplesSchema = z.array(
  z.object({ id: z.string(), has_reference: z.boolean() }),
);
export async function request<T>(
  path: string,
  schema: z.ZodType<T>,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  const body: unknown = await response.json();
  if (!response.ok) {
    const error = z.object({ detail: z.unknown() }).safeParse(body);
    throw new Error(
      error.success && typeof error.data.detail === "string"
        ? error.data.detail
        : `Request failed (${response.status})`,
    );
  }
  return schema.parse(body);
}
