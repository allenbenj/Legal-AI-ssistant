import React, { useEffect, useState } from "react";
import apiFetch from "../apiClient";
import { Plus, Edit } from "lucide-react";
import useLoadingState from "../hooks/useLoadingState";
import TableSkeleton from "./skeletons/TableSkeleton";

export interface WorkflowNode {
  id: string;
  type: string;
  label?: string;
}

export interface WorkflowConnection {
  from: string;
  to: string;
}

export interface Workflow {
  id: string;
  name: string;
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
}

const WorkflowDesigner: React.FC = () => {
  const { isLoading, error, data: workflows, executeAsync } =
    useLoadingState<Workflow[]>();
  const [selected, setSelected] = useState<Workflow | null>(null);

  useEffect(() => {
    executeAsync(() =>
      apiFetch("/api/v1/workflows").then((res) => {
        if (!res.ok) {
          throw new Error("Failed to load workflows");
        }
        return res.json();
      })
    );
  }, []);

  const editWorkflow = (wf: Workflow) => {
    setSelected(wf);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Workflow Designer</h2>
        <button className="px-3 py-2 bg-blue-600 text-white rounded-lg flex items-center gap-2 hover:bg-blue-700">
          <Plus className="w-4 h-4" />
          Create Workflow
        </button>
      </div>

      {selected && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="mb-4 font-medium">{selected.name}</div>
          {/* Workflow editing UI would go here */}
        </div>
      )}

      <div className="bg-white rounded-lg shadow">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold">Saved Workflows</h3>
        </div>
        <div className="p-6 space-y-4">
          {isLoading && <TableSkeleton rows={3} columns={2} />}
          {error && (
            <div className="text-red-600 space-x-2">
              <span>Error loading workflows</span>
              <button
                className="underline"
                onClick={() =>
                  executeAsync(() =>
                    apiFetch("/api/v1/workflows").then((res) => {
                      if (!res.ok) throw new Error("Failed to load workflows");
                      return res.json();
                    })
                  )
                }
              >
                Retry
              </button>
            </div>
          )}
          {workflows &&
            workflows.map((workflow) => (
              <div
                key={workflow.id}
                className="border rounded-lg p-4 flex items-center justify-between"
              >
                <div>
                  <div className="font-medium">{workflow.name}</div>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    className="p-2 hover:bg-gray-100 rounded"
                    onClick={() => editWorkflow(workflow)}
                  >
                    <Edit className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
};

export default WorkflowDesigner;
