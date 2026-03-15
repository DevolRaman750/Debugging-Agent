"use client";

import SideBar from "@/components/side-bar/SideBar";
import AuthGuard from "@/components/auth/AuthGuard";

export default function AuthenticatedLayout({
  children,
  isPublicRoute = false,
}: {
  children: React.ReactNode;
  isPublicRoute?: boolean;
}) {
  return (
    <AuthGuard isPublicRoute={isPublicRoute}>
      {isPublicRoute ? (
        children
      ) : (
        <div className="flex h-screen">
          <SideBar />
          <main className="flex-1 overflow-auto">{children}</main>
        </div>
      )}
    </AuthGuard>
  );
}
