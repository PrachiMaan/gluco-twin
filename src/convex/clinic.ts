import { getAuthUserId } from "@convex-dev/auth/server";
import { v } from "convex/values";

import { mutation, query } from "./_generated/server";

/** Alert keys acknowledged by the signed-in clinician for one patient. */
export const listAcks = query({
  args: { patientId: v.string() },
  handler: async (ctx, { patientId }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return [];
    const rows = await ctx.db
      .query("alertAcks")
      .withIndex("by_user_patient", (q) => q.eq("userId", userId).eq("patientId", patientId))
      .take(200);
    return rows.map((row) => row.alertKey);
  },
});

export const toggleAck = mutation({
  args: { patientId: v.string(), alertKey: v.string() },
  handler: async (ctx, { patientId, alertKey }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in to acknowledge alerts");
    const existing = await ctx.db
      .query("alertAcks")
      .withIndex("by_user_patient", (q) => q.eq("userId", userId).eq("patientId", patientId))
      .filter((q) => q.eq(q.field("alertKey"), alertKey))
      .first();
    if (existing) {
      await ctx.db.delete(existing._id);
      return false;
    }
    await ctx.db.insert("alertAcks", {
      userId,
      patientId,
      alertKey,
      acknowledgedAt: Date.now(),
    });
    return true;
  },
});

/** Short clinical notes attached to a patient by the signed-in clinician. */
export const listNotes = query({
  args: { patientId: v.string() },
  handler: async (ctx, { patientId }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) return [];
    const rows = await ctx.db
      .query("patientNotes")
      .withIndex("by_user_patient", (q) => q.eq("userId", userId).eq("patientId", patientId))
      .order("desc")
      .take(20);
    return rows.map((row) => ({ id: row._id, body: row.body, createdAt: row.createdAt }));
  },
});

export const addNote = mutation({
  args: { patientId: v.string(), body: v.string() },
  handler: async (ctx, { patientId, body }) => {
    const userId = await getAuthUserId(ctx);
    if (userId === null) throw new Error("Sign in to save notes");
    const trimmed = body.trim().slice(0, 400);
    if (!trimmed) return null;
    await ctx.db.insert("patientNotes", {
      userId,
      patientId,
      body: trimmed,
      createdAt: Date.now(),
    });
    return trimmed;
  },
});
