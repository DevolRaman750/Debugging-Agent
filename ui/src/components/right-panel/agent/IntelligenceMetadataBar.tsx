import React from "react";
import { AlertOctagon, AlertTriangle, Brain, CheckCircle2, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { IntelligenceMetadata } from "@/models/chat";

interface IntelligenceMetadataBarProps {
  metadata: IntelligenceMetadata;
}

function getConfidenceUI(confidence: IntelligenceMetadata["confidence"]) {
  if (confidence === "HIGH") {
    return {
      className: "bg-emerald-100 text-emerald-700 border-emerald-200",
      Icon: CheckCircle2,
    };
  }

  if (confidence === "MEDIUM") {
    return {
      className: "bg-amber-100 text-amber-700 border-amber-200",
      Icon: AlertTriangle,
    };
  }

  return {
    className: "bg-red-100 text-red-700 border-red-200",
    Icon: AlertOctagon,
  };
}

export default function IntelligenceMetadataBar({
  metadata,
}: IntelligenceMetadataBarProps) {
  const confidence = getConfidenceUI(
    (metadata.confidence as IntelligenceMetadata["confidence"]) || "LOW",
  );

  return (
    <div className="flex flex-row flex-wrap gap-2 items-center mb-2 text-xs">
      <Badge
        variant="outline"
        className={`inline-flex items-center gap-1.5 font-semibold ${confidence.className}`}
      >
        <confidence.Icon className="w-3 h-3" />
        {metadata.confidence}
      </Badge>

      {metadata.pattern_matched ? (
        <span className="inline-flex items-center rounded-full border border-indigo-200 text-indigo-700 bg-indigo-50 px-2 py-0.5">
          {metadata.pattern_matched}
        </span>
      ) : null}

      {metadata.fast_path ? (
        <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-yellow-700 bg-yellow-50 border border-yellow-200">
          <Zap className="w-3 h-3" />
          Fast Path
        </span>
      ) : (
        <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-violet-700 bg-violet-50 border border-violet-200">
          <Brain className="w-3 h-3" />
          Deep Analysis
        </span>
      )}

      <span className="text-muted-foreground">{metadata.processing_time_ms}ms</span>

      {metadata.causes_found > 1 ? (
        <span className="text-muted-foreground">{metadata.causes_found} causes found</span>
      ) : null}
    </div>
  );
}
