import pulumi
import pulumi_azure_native as azure_native
import pulumi_docker as docker

config = pulumi.Config()
resource_group_name = config.get("resource-group-name") or "rg-training-notebooks"

# Create an Azure Resource Group
resource_group = azure_native.resources.ResourceGroup(
    "resource-group",
    resource_group_name=resource_group_name,
)

# Create an Azure Container Registry
registry = azure_native.containerregistry.Registry(
    "registry",
    resource_group_name=resource_group.name,
    sku=azure_native.containerregistry.SkuArgs(
        name=azure_native.containerregistry.SkuName.BASIC,
    ),
    admin_user_enabled=True,
)

# Get registry credentials
credentials = pulumi.Output.all(resource_group.name, registry.name).apply(
    lambda args: azure_native.containerregistry.list_registry_credentials(
        resource_group_name=args[0],
        registry_name=args[1],
    )
)

admin_username = credentials.apply(lambda c: c.username)
admin_password = credentials.apply(lambda c: c.passwords[0].value)

image_name = registry.login_server.apply(lambda server: f"{server}/training-notebooks:latest")

# Build and push the Docker image to ACR
image = docker.Image(
    "image",
    image_name=image_name,
    build=docker.DockerBuildArgs(
        context="../..",
        dockerfile="../../deploy/Dockerfile",
        platform="linux/amd64",
    ),
    registry=docker.RegistryArgs(
        server=registry.login_server,
        username=admin_username,
        password=admin_password,
    ),
)

# Create an Azure Container Instance
container_group = azure_native.containerinstance.ContainerGroup(
    "container-group",
    resource_group_name=resource_group.name,
    os_type=azure_native.containerinstance.OperatingSystemTypes.LINUX,
    restart_policy=azure_native.containerinstance.ContainerGroupRestartPolicy.ON_FAILURE,
    image_registry_credentials=[
        azure_native.containerinstance.ImageRegistryCredentialArgs(
            server=registry.login_server,
            username=admin_username,
            password=admin_password,
        )
    ],
    ip_address=azure_native.containerinstance.IpAddressArgs(
        type=azure_native.containerinstance.ContainerGroupIpAddressType.PUBLIC,
        ports=[
            azure_native.containerinstance.PortArgs(
                port=8888,
                protocol=azure_native.containerinstance.ContainerGroupNetworkProtocol.TCP,
            ),
        ],
        dns_name_label="training-notebooks",
    ),
    containers=[
        azure_native.containerinstance.ContainerArgs(
            name="jupyter",
            image=image.image_name,
            resources=azure_native.containerinstance.ResourceRequirementsArgs(
                requests=azure_native.containerinstance.ResourceRequestsArgs(
                    cpu=4.0,
                    memory_in_gb=8.0,
                ),
            ),
            ports=[
                azure_native.containerinstance.ContainerPortArgs(
                    port=8888,
                ),
            ],
            environment_variables=[
                azure_native.containerinstance.EnvironmentVariableArgs(
                    name="PROVIDER",
                    value="local",
                ),
                azure_native.containerinstance.EnvironmentVariableArgs(
                    name="OLLAMA_BASE_URL",
                    value="http://localhost:11434",
                ),
            ],
        ),
    ],
)

notebook_url = pulumi.Output.all(container_group.ip_address).apply(
    lambda args: f"http://{args[0].fqdn}:8888" if args[0] and args[0].fqdn else "URL not yet available"
)

pulumi.export("notebook_url", notebook_url)
