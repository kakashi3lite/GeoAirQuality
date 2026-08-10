# Deploy — Kubernetes, Helm, nginx, Monitoring

All deployment assets consolidated under one roof.

| Directory | Contents |
|-----------|----------|
| [`k8s/`](k8s/) | Manifests: `api-deployment.yaml`, `pipeline-deployment.yaml`, `production-deployment.yaml` (namespace, PostGIS StatefulSet, Redis, HPA, Ingress, PDBs, NetworkPolicy) |
| [`helm/`](helm/) | Helm chart (`api/chart` → here) |
| [`nginx/`](nginx/) | Reverse-proxy config + TLS dir |
| [`monitoring/`](monitoring/) | Prometheus scrape config + Grafana provisioning |

## Quick reference

```bash
# Apply production manifests
kubectl apply -f deploy/k8s/production-deployment.yaml

# Helm
helm install geoairquality deploy/helm --values deploy/helm/values-prod.yaml
```

## SECURITY

`deploy/k8s/production-deployment.yaml` contains a **placeholder Secret**
(`CHANGE_THIS_*`). Never commit real credentials — use SealedSecrets or
the External Secrets Operator. See [`docs/guides/security.md`](../docs/guides/security.md).
