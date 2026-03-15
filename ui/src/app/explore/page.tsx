"use client";

import React from "react";
import SideBar from "@/components/side-bar/SideBar";
import { ExploreProvider } from "@/hooks/useExploreContext";
import ExploreContent from "@/components/explore/ExploreContent";

export default function ExplorePage() {
  return (
    <ExploreProvider>
      <div className="flex h-screen w-full bg-background text-foreground">
        <SideBar />
        <ExploreContent />
      </div>
    </ExploreProvider>
  );
}
