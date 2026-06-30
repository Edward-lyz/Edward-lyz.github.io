# 【工程】Docker 和 Kubernetes 中的 Linux 容器特权详解

# 引言：Linux Capabilities 与容器特权概述

传统上，在 Linux 中只有超级用户 (UID=0，即 root) 拥有完整的系统权限，而非特权用户受限于正常的权限检查。从 Linux 2.2 开始，引入了 **Capabilities** 机制，将 root 的特权拆分成数十种更细粒度的权限位。这样可以按照“最小权限原则”赋予进程所需的最低特权：应用可以仅拥有完成任务所需的部分“root”权限，而不必拥有全部超级用户权限。在容器环境中，Capabilities 是安全隔离的重要基础：容器进程通常以 root 身份运行但默认只保留有限的 Capabilities，从而降低其破坏力。Kubernetes 和 Docker 均利用该机制，通过配置允许或剥夺某些 Capabilities，控制容器内程序的权限。

下面将详细列出常见的 Linux Capabilities，以及它们的含义、在 Linux 内核中的作用，及在容器中的影响和潜在风险。

# 常见 Linux Capabilities 列表及含义

Linux 内核目前实现了大约 40 个不同的 Capabilities。下表按类别汇总了主要的 Capabilities，每项包含其含义/作用以及在容器环境中的影响或风险。

## 文件系统与文件权限相关 Capabilities

| Capability | Linux 含义及权限 | 在容器中的影响与风险 |
| --- | --- | --- |
| **CAP_CHOWN** | 允许对文件的所有者 UID 和 GID 进行任意更改。 | **影响：**容器进程可随意更改文件属主，绕过所有者检查。这在容器内意味着应用可以修改任何文件的拥有者。如果挂载了主机目录，共享卷内的文件所有权也可能被修改，造成主机文件权限紊乱。 |
| **CAP_DAC_OVERRIDE** | 绕过文件读、写和执行的权限检查（DAC，自主访问控制）。 | **影响：**容器进程可无视文件的 Unix 权限位进行读写操作。例如，本来无权限读取的文件在有此 Capability 时可被读取。若攻击者获取容器控制权，可能访问或篡改原本受限的文件内容。 |
| **CAP_DAC_READ_SEARCH** | 绕过文件只读检查，以及目录的读取/执行权限检查。还允许使用 `open_by_handle_at(2)` 等特殊文件访问调用。 | **影响：**容器进程可在没有读权限的情况下读取文件或目录列表。这增加了信息泄露风险，攻击者能遍历或读取本应不可见的目录结构和文件。 |
| **CAP_FOWNER** | 绕过某些与文件所有权相关的权限检查（例如修改文件权限`chmod`、修改时间戳`utime`），即使当前进程并非该文件所有者。还允许忽略目录粘滞位删除文件等。 | **影响：**容器内进程可修改任意文件的权限和属性（除了另有专门 Cap 控制的情形），即使不拥有该文件。这可能被利用来篡改应用文件、提升权限（如将可执行文件设置为 Setuid）。在共享卷场景下，可能影响主机上的文件属性。 |
| **CAP_FSETID** | 文件被修改时不清除其 Set-UID/Set-GID 位；允许在不匹配文件当前 GID 的情况下设置文件的组ID位。 | **影响：**容器进程修改带 Setuid/Setgid 的程序时，可保留这些特权位，从而在容器内维持提权机制。同时也可在创建文件时设置任意组ID位。可能被攻击者用于维持权限提升的后门（如保留Setuid位的恶意程序）。 |
| **CAP_MKNOD** | 允许使用 `mknod(2)` 创建设备特殊文件（如块设备、字符设备）。 | **影响：**容器内进程可创建新的设备文件节点。如果容器还拥有对相应设备的访问权限（受 cgroup 设备控制），则可能直接与主机硬件交互，带来安全隐患。通常默认情况下容器的 cgroup 设备策略会阻止访问未授权设备，但赋予 CAP_MKNOD仍是高风险操作。 |
| **CAP_LINUX_IMMUTABLE** | 可设置文件不可变标志和追加写入标志（`FS_IMMUTABLE_FL`, `FS_APPEND_FL`）。 | **影响：**容器进程可将文件标记为不可变或只能追加，改变文件系统行为。例如攻击者可将关键配置文件标记不可变以阻止修改，或者恶意程序利用此限制日志只能追加无法删除。虽然这不直接破坏隔离，但可能被用来妨碍运维或取证。 |
| **CAP_LEASE** | 可对任意文件设置文件租约（leases）。文件租约允许进程监视对文件的打开/关闭等事件。 | **影响：**容器进程可对文件施加租约，从而控制文件访问的同步。这主要用于文件锁定机制，本身危害不大。但恶意程序可能利用租约阻塞其他进程对某文件的访问（造成DoS），或监视宿主对共享文件的操作。 |
| **CAP_SETFCAP** | 允许对文件设置 **文件Capabilities**（即给可执行文件附加特权位）。 | **影响：**容器内可通过为可执行文件设置特殊Capabilities来提升该程序执行时的权限。例如给二进制文件赋予网络或管理特权。通常Docker镜像构建时会去除文件上的Capabilities扩展属性，但若容器运行时具有CAP_SETFCAP，攻击者可能自行赋予文件额外特权，从而在下次运行时获得更高权限。 |

*解读：* 上述 Capabilities 涉及对文件和文件系统权限的控制，在容器中默认 **应尽量减少授予**。例如，Docker 默认启用 CAP_CHOWN 等基本文件操作能力，但像 CAP_DAC_OVERRIDE 这类可绕过权限检查的能力默认是**不赋予**容器的，以防止容器绕过文件系统的隔离限制。即便如此，哪怕基本的文件操作权限，也可能在共享卷场景下被利用对宿主文件造成影响，因此需要根据实际需求谨慎添加或放权。

## 用户和进程相关 Capabilities

| Capability | Linux 含义及权限 | 在容器中的影响与风险 |
| --- | --- | --- |
| **CAP_SETUID** | 允许对进程的用户ID进行任意修改（例如调用`setuid(2)`将进程切换为任意UID）。 | **影响：**容器内进程可将自身切换为任意用户，包括提升为root或降为其他用户。这对非root进程提权有用。但在容器内由于默认即为root用户启动，此Capability主要影响从root降权的灵活性。如果攻击者获取非root的容器进程并拥有CAP_SETUID，可切换为root，提升在容器内的权限。 |
| **CAP_SETGID** | 允许对进程的组ID和辅助组列表进行任意修改。 | **影响：**类似CAP_SETUID，容器进程可更改其组身份。攻击者利用它可加入受限制的组以获取额外权限。不过默认容器即root组，此能力更多用于在容器内做细粒度权限管理。在恶意情形下，可配合CAP_SETUID实现进程身份完全伪装。 |
| **CAP_SETPCAP** | 允许修改进程的Capability集合，例如向自身继承集添加目前 bounding set 中允许的Capabilities，或从 bounding set 中移除Capabilities。 | **影响：**容器内进程可调整自身及子进程的特权位。如果容器运行时 bounding set 限制了某些 Cap，这个能力通常也无法突破该限制（Linux 内核默认不允许进程将不在边界集的Capability添加回来）。因此该Capability主要用于_降低_权限。但在不常见的配置下，可能被利用来重新启用某些被临时移除的特权。一般容器不需要授予此Capability。 |
| **CAP_KILL** | 绕过信号发送的权限检查，允许向任意进程发送信号（包括向不属于同一用户的进程发送`kill`信号）。 | **影响：**容器内进程可向容器中的其他进程发送信号（SIGKILL等）而不需考虑UID是否相同。这对容器自身内部的进程管理有用（Docker 默认保留该能力）。但默认情况下容器的进程隔离(pid namespace)使其看不到宿主或其他容器的进程，所以CAP_KILL影响局限于容器内部。如果禁用了PID隔离（如Kubernetes的`hostPID: true`模式），CAP_KILL将允许容器向宿主上任意进程发信号，这是严重的权限泄露。 |
| **CAP_SYS_PTRACE** | 允许跟踪（调试）任意进程，即使用`ptrace`对其他进程进行检查和操作。也允许通过`/proc`读取其他进程的敏感信息（如内存、文件描述符）。 | **影响：**默认容器启用PID命名空间，因此即使有CAP_SYS_PTRACE也只能调试容器内的进程。这仍可能带来风险：恶意程序可窥探同容器其他进程的内存（窃取机密信息）或注入代码。如果容器与宿主共享PID命名空间（非常不建议），那么拥有CAP_SYS_PTRACE的容器进程可以调试宿主上的进程，达到完全控制和数据窃取的程度。因此除非有调试需求，应避免容器获取此Capability。 |
| **CAP_SYS_NICE** | 允许调整进程调度优先级和调度策略，包括降低任意进程的nice值（提高优先级）或将进程设为实时调度等。也允许将任意进程绑定CPU亲和性、内存节点等（迁移进程到其他CPU/NUMA节点）。 | **影响：**容器内拥有此能力可使进程提高自己的或他进程的优先级，从而独占更多CPU（可能影响同容器其他服务）。如果一台宿主上多个容器共享CPU，某容器滥用CAP_SYS_NICE可能导致CPU饥饿攻击。不过通常各容器间CPU隔离靠cgroups控制，此Capability影响主要在容器内部。除实时应用需要，一般不应赋予容器该能力。 |
| **CAP_SYS_RESOURCE** | 允许覆盖资源限制和配额，例如绕过`ulimit`/RLIMIT约束（增加进程最大文件句柄数、内存锁定大小等），超越磁盘配额限制，保留ext文件系统的预留空间等。还能提高POSIX消息队列配额、调低进程的 OOM 分数（避免被OOM杀死）等。 | **影响：**容器进程可突破分配给它的某些资源限制。例如，若容器被设置了文件句柄或内存限制，有CAP_SYS_RESOURCE时可自行提高这些限制。这可能导致容器消耗超过预期的资源甚至影响宿主稳定性（如耗尽主机文件描述符表）。因此默认容器不应具有此能力。Kubernetes 和 Docker 通常通过 cgroups 实施资源限制，CAP_SYS_RESOURCE的滥用可绕过部分cgroup限制（如进程数），带来DoS风险。 |
| **CAP_SYS_CHROOT** | 允许调用`chroot(2)`改变进程根目录，并在进入其他 mount namespace 时无需额外权限。 | **影响：**容器内进程可自行对自身执行chroot操作。容器本身就是一种chroot环境，因此此能力通常作用不明显。大多数应用无需在容器内再次chroot，因此可考虑去除。攻击者获得该能力也难以突破容器，因为仍受限于容器的文件系统命名空间。 |
| **CAP_SYS_PACCT** | 允许启用或禁用进程计帐 (`acct(2)`)，即开启/关闭系统级的进程资源统计功能。 | **影响：**容器内几乎不会用到开启宿主级进程计帐的功能（宿主通常不允许容器直接操作全局计帐）。即使赋予此能力，若容器有独立的 PID 和文件系统命名空间，它也只能影响容器内部（但通常计帐是全局的，容器可能无法真正执行）。此Capability风险较低但无必要，默认应移除。 |

*解读：* 以上涉及**进程管理**的 Capabilities 在容器中一般也不是默认全部授予的。Docker 默认仅保留其中的少数（例如 CAP_KILL）。大部分如 CAP_SYS_PTRACE、CAP_SYS_NICE 等都不在默认集内，因为它们对系统其他进程或资源有较大影响。如果容器需要这些能力（如需在容器内调试程序，则需要 CAP_SYS_PTRACE），应在**严格限制范围**的情况下按需添加，并确保隔离（如不要与宿主共享 PID namespace），以免造成越权访问。

## 网络相关 Capabilities

| Capability | Linux 含义及权限 | 在容器中的影响与风险 |
| --- | --- | --- |
| **CAP_NET_BIND_SERVICE** | 允许将套接字绑定到特权网络端口（端口号<1024）。 | **影响：**容器进程可以监听低号端口（例如80, 443）而无需提升为宿主机上的root。这对运行Web服务器容器很有用（Docker 默认保留此能力）。安全风险相对较小，因为仅影响容器自身网络命名空间。不过，如不需要低端口监听，可移除此能力来迫使应用使用高端口映射，从而降低潜在滥用。 |
| **CAP_NET_BROADCAST** | *（已废弃）* 允许进行网络广播和多播监听。 | **影响：**该能力在现代内核中未被使用，Docker/K8s 中通常无效。因此对容器无实际影响，可以忽略。 |
| **CAP_NET_ADMIN** | 允许执行多种网络相关管理操作，例如配置网络接口、管理路由表、防火墙规则、启用侦听模式等。 | **影响：**容器进程可对网络进行广泛控制。例如，它可以修改容器内网络接口IP、路由和iptables规则等。**正常情况下容器网络与宿主隔离**，所以这些操作仅作用于容器自身网络命名空间。但这仍存在风险：恶意容器可通过修改路由/iptables来拦截或篡改经过容器的流量。如果容器使用主机网络（Docker `--network=host` 或 K8s `hostNetwork: true`），那么CAP_NET_ADMIN将直接影响宿主网络设置，包括更改主机接口配置、防火墙等，严重危及主机网络安全。因此非必要不应赋予容器该权限。 |
| **CAP_NET_RAW** | 允许使用RAW套接字和PACKET套接字，进行底层网络通信；也允许绑定网络接口用于透明代理。 | **影响：**具备此能力，容器可构造原始网络包（例如发送自定义IP包，执行ARP欺骗等）。Docker 默认保留该能力以支持ping等工具运行。在容器网络隔离情况下，RAW套接字流量通常局限在容器自身网络namespace内。然而恶意容器仍可利用它进行网络扫描、流量嗅探甚至发起基于IP包的攻击。如果容器网络与宿主未隔离，风险更高。所以对不需要底层网络访问的应用，可考虑去除CAP_NET_RAW。 |

*解读：* 网络类 Capabilities 默认配置上 **相对保守**。Docker 默认允许 NET_BIND_SERVICE 和 NET_RAW，以支持常见网络服务和Ping等操作，但**不赋予**NET_ADMIN，因为后者权限过大。对于需要复杂网络配置的容器（例如运行自定义路由/负载均衡软件），可能需要 CAP_NET_ADMIN，但应确保容器网络隔离良好，避免影响其他容器或宿主。原则上，容器应仅拥有其职能所需的最低网络权限。

## IPC 与内存相关 Capabilities

| Capability | Linux 含义及权限 | 在容器中的影响与风险 |
| --- | --- | --- |
| **CAP_IPC_LOCK** | 允许锁定内存，防止内存页被交换到磁盘（使用`mlock`, `mlockall`等），并允许使用大页内存分配。 | **影响：**容器进程可将内存页锁定在RAM中，增大其常驻内存占用。例如数据库或缓存应用可能需要此权限来防止关键数据被swap出去。如果滥用，恶意进程可以锁定大量内存而不受swap缓解，可能导致宿主内存压力增大甚至耗尽。幸好cgroups内存限制通常仍对其生效，但没有CAP_IPC_LOCK时调用mlock会失败。因此仅当应用明确需要（如Memcached需要锁定内存）才添加此Capability。 |
| **CAP_IPC_OWNER** | 绕过对 **System V IPC 对象**（如共享内存段、信号量）的权限检查。 | **影响：**容器进程可访问/控制所有的 SysV IPC 资源，即使并非它们的创建者。在启用了 IPC Namespace 的容器中，该影响局限于容器内部的IPC对象；但如果容器与宿主共用IPC命名空间（很少见的配置），攻击者可访问宿主的IPC对象，造成信息泄露或干扰。通常容器不需要直接操作宿主的SysV IPC，因此应避免授予此能力。 |

*解读：* 这两项Capability涉及**进程间通信和内存**。大部分应用容器默认**不具有**这两项特权，因为它们并非通用需求，而且滥用可能导致对宿主资源的影响（内存锁定导致宿主内存紧张等）。如果需要使用诸如 memlock 的功能，应在评估风险后在容器运行时通过`--cap-add IPC_LOCK`显式开启，并搭配内存限额以防止滥用。

## 系统管理与硬件相关 Capabilities

| Capability | Linux 含义及权限 | 在容器中的影响与风险 |
| --- | --- | --- |
| **CAP_SYS_ADMIN** | *（系统管理“万能”权限）* 此Capability涵盖广泛的系统管理操作，包括挂载和卸载文件系统、调用`pivot_root`改变根、启用交换、设置主机名/域名等。它还隐含许多子功能，例如**所有**新引入的 CAP_BPF、CAP_PERFMON、CAP_CHECKPOINT_RESTORE 等在没有独立赋予时也被CAP_SYS_ADMIN覆盖。由于范围极广，CAP_SYS_ADMIN被称为“新的root权限”。 | **影响：这是容器中最危险的Capability**。赋予容器CAP_SYS_ADMIN几乎等同于赋予其root权限，可以进行大量特权操作（如挂载宿主文件系统、访问设备、进行系统调试等）。实践表明，仅有CAP_SYS_ADMIN的容器便可利用内核接口实施逃逸。例如，攻击者曾利用具有CAP_SYS_ADMIN的容器挂载宿主的cgroup文件系统并结合`notify_on_release`机制取得宿主shell。总之，在**非特权容器**场景下一定要避免给容器此权限；如果必须使用，应将容器视为全权限并采取隔离措施（如独立主机运行）。 |
| **CAP_SYS_MODULE** | 允许加载和卸载内核模块（`init_module`, `delete_module`）。 | **影响：**容器可向宿主内核加载任意模块或卸载现有模块，等同于对内核植入代码。这是非常高危的操作：攻击者可加载恶意内核模块取得对宿主的完全控制。因此此Capability绝不应赋予普通容器。在需要时通常通过`--privileged`（授予所有能力）来运行特殊容器，例如硬件驱动容器，但这意味着放弃隔离。 |
| **CAP_SYS_RAWIO** | 允许执行原始I/O操作，包括直接访问硬件I/O端口（iopl/ioperm）、访问`/dev/mem`和`/dev/kmem`（物理内存）、读取内核内存映像`/proc/kcore`等。 | **影响：**具备此能力的容器进程可以绕过操作系统提供的设备文件接口，直接对内存和硬件进行读写。这意味着攻击者可读取整个主机内存内容或修改硬件寄存器等，几乎可完全控制主机。这在容器环境下是**极其危险**和不必要的，默认绝不会赋予容器。如需访问特定设备，应通过设备节点挂载和cgroup设备白名单控制，而不开放SYS_RAWIO。 |
| **CAP_SYS_TIME** | 允许设置系统时钟和实时硬件时钟。 | **影响：**如果容器没有独立时间命名空间（当前大多数容器和宿主**共用系统时间**），则赋予该能力的容器可以更改宿主的系统时间。这可能导致日志时间篡改、安全协议失效（例如TLS证书验证依赖时间）等严重问题。即使将来时间命名空间隔离，此能力影响也应谨慎对待，因为不正确的时间可能扰乱分布式系统行为。一般容器不需要修改系统时间，除非运行时间同步服务，此时应隔离容器使之不会影响宿主。 |
| **CAP_SYS_TTY_CONFIG** | 允许配置TTY设备，例如使用`vhangup(2)`挂起一个终端，以及对虚拟终端执行某些特权`ioctl`操作。 | **影响：**容器进程可操作系统的TTY/控制台，例如强制注销某终端的会话。这能力主要对宿主控制台有意义。在容器内，其用途很有限，风险也不高。但若容器共享了宿主的TTY设备（极少情况），攻击者可能利用它干扰宿主的终端。通常无需赋予容器该权限。 |
| **CAP_SYS_BOOT** | 允许调用`reboot(2)`重新启动系统或`kexec_load(2)`加载新内核。 | **影响：**如果容器具有此能力并能访问适当的设备接口，它可以直接触发宿主机重启或替换内核！这对宿主而言是毁灭性的（拒绝服务甚至持久控制）。因此除非是在受控环境中运行特权系统管理容器，否则绝不应赋予。这一权限通常只由系统初始化进程或管理员进程在宿主上使用。 |
| **CAP_SYSLOG** | 允许执行特权的`syslog(2)`操作（例如读取或清除内核环缓冲区），以及在`kptr_restrict`设置为1时读取内核地址等敏感信息。 | **影响：**容器可读取内核日志、提取潜在敏感的内核地址信息，或调整内核日志行为。这可能泄露宿主内核的信息（有助于攻击者利用漏洞）。默认容器不具有此能力。在大多数情况下不需要授予容器对宿主内核日志的控制权限。 |

*解读：* 这一组Capbilities涉及**系统级的管理控制**，通常只有宿主管理员才需要。Docker/Kubernetes 默认**完全禁止**容器使用这些能力，因为它们几乎都可以用于突破容器隔离或直接控制宿主。特别是CAP_SYS_ADMIN，它覆盖功能最多，也是历次容器逃逸漏洞中**最常被利用**的权限。实践经验表明，如果发现容器需要上述某项能力，很可能意味着该容器应该提升为“特权容器”而在隔离环境下运行，而不应轻易在多租户环境下赋予单一Capability。总之，普通应用容器不应具备这些系统管理特权。

## 安全审计与系统监控 Capabilities

| Capability | Linux 含义及权限 | 在容器中的影响与风险 |
| --- | --- | --- |
| **CAP_AUDIT_WRITE** | 允许将记录写入内核审计日志（例如`auditd`守护进程使用）。 | **影响：**容器可产生审计日志记录。Docker 默认保留该能力以方便审计日志工作。风险在于恶意容器可能生成大量伪造审计日志骚扰管理员或掩盖自身行为。不过审计日志通常针对宿主，全容器环境下审计可能未启用。可按需去除该能力来减少不必要的日志访问。 |
| **CAP_AUDIT_READ** | 允许通过多播 Netlink 套接字读取内核审计日志。 | **影响：**容器可以读取宿主的安全审计日志内容。这可能泄露其他容器或宿主的敏感操作记录，因而安全风险较高。默认容器不应有此权限，除非你有专门的安全监控容器需要汇总审计事件且可信任。 |
| **CAP_AUDIT_CONTROL** | 允许启用/禁用内核审计、修改审计规则、读取审计设置等。 | **影响：**容器可更改宿主的审计策略，甚至关闭审计子系统，从而令安全监控失效。此Capability应仅宿主上的安全守护进程使用，**决不能**赋予普通容器，否则攻击者可以隐藏其入侵痕迹。 |
| **CAP_MAC_ADMIN** | 允许修改MAC（强制访问控制）配置和状态。例如调整SELinux或AppArmor策略。 | **影响：**容器可更改宿主的强制访问控制策略（如SELinux策略装载卸载、标签改动）。这将直接破坏容器之间以及容器对宿主的安全隔离机制。显然不应赋予容器该权限，否则等同赋予其修改安全机制的钥匙。 |
| **CAP_MAC_OVERRIDE** | 允许绕过强制访问控制（MAC）检查。例如无视SELinux/AppArmor策略强制访问受限资源。 | **影响：**容器进程可忽略宿主的MAC限制，访问本应被禁止的文件或资源。这同样严重破坏隔离。例如在启用SELinux的系统上，容器进程本来受`container_t`类型限制，若有MAC_OVERRIDE则可读取任意类型的文件。因此该能力绝不可授予容器。 |
| **CAP_BPF** | 允许执行特权级的BPF (Berkeley Packet Filter) 操作，例如加载BPF程序、修改BPF映射等。此能力在Linux 5.8中加入，原本属于CAP_SYS_ADMIN的BPF功能被独立出来。 | **影响：**容器可加载高级BPF程序到内核中。虽然内核对BPF程序有校验，但特权BPF仍可用于监视系统活动、修改网络包流程，甚至利用漏洞实现提权。如果赋予容器CAP_BPF，相当于允许其动态扩展内核功能，安全审计难度加大。除非运行eBPF工具容器（如性能分析）且充分信任，否则不应赋予此权限。 |
| **CAP_PERFMON** | 允许使用高性能计数器和跟踪机制，包括调用`perf_event_open`和使用部分具有性能影响的BPF操作。该能力在Linux 5.8中加入，用于从CAP_SYS_ADMIN中分离性能监控功能。 | **影响：**容器可使用`perf`性能分析接口监视系统（包括宿主和其他容器）的性能事件，例如CPU缓存命中、分支预测失误等。这可能间接泄露其他进程的执行特征（侧信道攻击）。因此默认不应给容器此能力，除非用于受控的性能监控容器。 |
| **CAP_CHECKPOINT_RESTORE** | 允许执行进程检查点/恢复相关的操作，例如修改`/proc/sys/kernel/ns_last_pid`、使用`clone3`的`set_tid`功能、读取其他进程的`/proc/[pid]/map_files`链接内容等。Linux 5.9加入，用于从CAP_SYS_ADMIN中分离CRIU（Checkpoint/Restore in Userspace）的特权。 | **影响：**容器可读取其他进程的内存映射信息并在一定程度上控制进程ID分配等。这主要服务于容器内进行进程迁移/恢复。如果恶意利用，可能泄露系统进程内存布局或制造进程ID碰撞。在一般应用容器中几乎用不到，除非部署CRIU容器进行容器迁移。因此常规场景不应赋予。 |
| **CAP_WAKE_ALARM** | 允许触发唤醒系统的闹钟（设置 `CLOCK_REALTIME_ALARM` 和 `CLOCK_BOOTTIME_ALARM` 定时器）。 | **影响：**容器可设置系统闹钟定时唤醒休眠的宿主。这在宿主不希望被唤醒时会造成影响（如笔记本休眠被唤醒）。在服务端场景影响不大，但通常容器无需关心宿主休眠管理。可根据需要决定赋予与否，安全风险较小。 |
| **CAP_BLOCK_SUSPEND** | 允许使用阻止系统挂起的功能，例如`epoll`的`EPOLLWAKEUP`标志、操作`/proc/sys/wake_lock`等，以阻止进入休眠。 | **影响：**容器可以阻止宿主机进入休眠状态。这对某些需要持续运行的任务有用，但若被滥用会导致宿主无法节能休眠。在服务器场景通常无关紧要（服务器不休眠），安全风险低。但在桌面/嵌入式环境下应注意不要让不可信容器持有此能力，以免影响电源管理。 |

*解读：* 上述为**安全、审计及系统监控**相关的 Capabilities。一般来说，这些权限要么是给安全工具使用（如审计日志），要么是细分自 CAP_SYS_ADMIN 的特殊权限。在默认容器环境下，几乎**全部都不需要**。Docker 默认仅保留了 AUDIT_WRITE 供记录日志，但大多数场景下可去除而不影响应用运行。像 MAC_ADMIN/OVERRIDE、AUDIT_CONTROL 这类破坏宿主安全策略的能力更是从不该出现在容器中。一些新能力如 BPF、PERFMON 则视为高级调试/监控用途，除非明确需要，否则不应赋予容器。总之，这一类别的Capabilities通常只有在**特定场景**下由高度信任的容器使用，其余情形下应维持默认禁用状态。

---

以上我们按类别列出了 Linux Capabilities 及其意义和影响。总结来说，Linux Capabilities 将原本属于root用户的特权拆解开来，使我们可以细粒度地控制容器的权限边界。默认情况下，Docker/容器运行时对容器**启用了少量必需的 Capabilities，其余的均被移除**。这样设计的目的，是尽量降低容器突破隔离的风险，同时又保留常见操作所需的权限。下一节我们将讨论在 Docker 和 Kubernetes 中如何配置和管理这些 Capabilities，包括 *privileged* 模式的含义。

# Docker 中的特权配置与 Capabilities 管理

Docker 在启动容器时会应用默认的能力集合，并提供参数让用户调整这些权限。**Docker 默认**采用“白名单”策略：**仅保留约 14 项常用且低风险的 Capabilities，其余全部移除**。上述默认保留的能力包括：`CAP_CHOWN`, `CAP_DAC_OVERRIDE`, `CAP_FOWNER`, `CAP_FSETID`, `CAP_MKNOD`, `CAP_NET_BIND_SERVICE`, `CAP_NET_RAW`, `CAP_SETGID`, `CAP_SETUID`, `CAP_SETFCAP`, `CAP_SETPCAP`, `CAP_SYS_CHROOT`, `CAP_KILL`, `CAP_AUDIT_WRITE`。这一列表正是前文提及的大多数基础文件操作和少数必要的网络/信号权限。

在 Docker 中，可以通过以下方式管理容器的 Capabilities：

- **显式增加单个 Capability：** 使用 `--cap-add` 标志。例如，`docker run --cap-add=NET_ADMIN ...` 将在容器进程的允许集合中增加 CAP_NET_ADMIN。可重复使用该标志添加多个不同能力。
- **显式移除单个 Capability：** 使用 `--cap-drop` 标志。例如，`docker run --cap-drop=MKNOD ...` 将从默认集删除 CAP_MKNOD。建议在不影响应用的情况下尽可能删减默认能力，例如 Red Hat 建议默认可去除 `AUDIT_WRITE`, `MKNOD`, `SETFCAP`, `SETPCAP` 几项。
- **移除所有再按需添加：** `--cap-drop ALL` 可以在启动容器时去除**全部** Capabilities，然后通过 `--cap-add` 添加回需要的一两个。例如，只保留修改系统时间的能力可用：`docker run --cap-drop ALL --cap-add=SYS_TIME ...`。这种方式实现了最小权限原则。
- **特权模式 (Privileged)：** 使用 `--privileged` 选项启动容器，则 Docker 将**赋予容器进程几乎所有的 Linux Capabilities**，并移除大部分隔离限制，使其与宿主几乎等同。具体而言，privileged 模式下容器获得所有 Capabilities，拥有对所有主机设备的访问权限，AppArmor/SELinux 型限制也大多解除，从而容器内的 root 就是宿主的 root。这一模式通常用于需要深度访问宿主系统的场景（如运行 Docker-in-Docker、特权调试容器等），**但安全风险极高**，应谨慎使用。在大多数情况下，通过细粒度 `--cap-add` 即可满足需求，而不必整个容器特权化。

Docker 的这些选项让管理员能够根据应用需求调整容器权限边界。需要注意，Capability的更改只在容器内生效，并不改变映像本身。当我们通过上述参数运行容器时，Docker 在创建容器进程前设置好其允许和有效能力集（Bounding/Permitted/Effective sets）。**Docker 在内核中实现了Capability bounding set，确保容器进程无法获取未被允许的能力**。例如，即使容器内有 CAP_SETPCAP 也无法添加超出 bounding set 之外的能力，这一机制保证了我们在启动时的配置是最终上限。

值得一提的是，Docker 默认还启用了其他安全机制配合 Capabilities 使用（详见后文），如：启用 Seccomp 默认配置过滤危险系统调用、应用默认的AppArmor配置文件等。这些默认措施进一步缩小了即使在授予某些Capabilities情况下容器可能造成的危害范围。

**示例：** 如果我们希望运行一个需要绑定低端口和调整网络设置的容器，可以执行：

`docker run -d --cap-add NET_ADMIN --cap-add NET_BIND_SERVICE nginx`

这样容器将拥有 CAP_NET_ADMIN 和 CAP_NET_BIND_SERVICE，以便既能修改自身网络接口又能监听80端口；而没有赋予例如 CAP_SYS_ADMIN 等无关高权能力。相反，如果直接使用 `--privileged`，上述目标也能达成，但容器还额外获得了许多不需要的危险权限，例如能够加载内核模块、挂载宿主文件系统等。因此应**优先选择最小权限**方案而非一刀切地使用特权模式。正如安全业界所建议的：“尽可能避免使用 CAP_SYS_ADMIN和 `--privileged`，除非万不得已”。

# Kubernetes 中的特权配置与 Capabilities 管理

在 Kubernetes 中，容器的权限控制通过 Pod 的 **SecurityContext** 来配置。每个 Pod 或容器可以在其 securityContext 中声明需要添加或移除的 Capabilities，以及是否以特权模式运行。关键配置项包括：

- **添加/移除 Capabilities：** 可以在容器的 `securityContext.capabilities` 下列出 `add` 和 `drop` 列表。例如：
    
    `apiVersion: v1 kind: Pod spec:   containers:   - name: myapp     image: myimage:latest     securityContext:       capabilities:         drop: ["ALL"]            # 先移除所有默认Capabilities         add: ["NET_ADMIN", "NET_RAW"]  # 添加所需的Capabilities`
    
    上述配置将使 `myapp` 容器去除默认所有能力，仅添加 CAP_NET_ADMIN 和 CAP_NET_RAW。Kubernetes 默认行为与Docker一致，即每个容器都有一套默认能力(与Docker默认14项相同)。通过 securityContext.capabilities，我们可以精细地调整这些权限。例如，将 `drop: ["ALL"]` 可以确保容器从零开始构建其权限集，需要什么加什么，从而达到**最小权限**。如果只想移除一两项默认Capability（例如出于安全考虑拿掉 `MKNOD`），也可以仅在drop列表列出它们。
    
- **特权容器 (privileged)：** 在 Pod 的 securityContext 中设置 `privileged: true` 会使对应容器以特权模式运行。例如：
    
    `apiVersion: v1 kind: Pod metadata:   name: privpod spec:   containers:   - name: privctr     image: ubuntu     securityContext:       privileged: true`
    
    这等效于 Docker 的 `--privileged`，即容器 `privctr` 将获得宿主几乎所有Capabilities和设备访问。Kubernetes的特权容器通常还需要配合设置 `hostPID`, `hostNetwork` 等才能完全等同宿主环境（如需要访问宿主的命名空间）。应当明确，**privileged:true 打开的容器具有极高权限风险**，应仅用于受信任环境下的运维任务（如运行特权 DaemonSet 来管理主机网络、存储）。
    

除了以上两点，Kubernetes SecurityContext 还有一些相关选项：

- **允许运行特权提升 (allowPrivilegeEscalation)：** 该选项默认随容器是否具有CAP_SYS_ADMIN或特权模式而定。K8s中，如果容器是privileged或有CAP_SYS_ADMIN，则allowPrivilegeEscalation会被**强制为true**（表示容器内进程可以通过`setuid`或文件提权等方式获取额外权限）。若希望即使容器内有一些Capabilities也禁止进程提权，可以将其设置false（需启用NoNewPrivileges支持）。一般使用Pod安全策略来禁止不必要的提权。
- **其他安全上下文**：如 `readOnlyRootFilesystem`, `runAsUser` 等，也在一定程度上影响容器权限。例如强制只读根文件系统可缓解某些CAP_SYS_ADMIN滥用挂载的风险；以非root用户运行容器可降低即使获取Capabilities也造成的破坏面。这些与Capabilities共同组成全面的容器安全上下文。

总之，在Kubernetes中我们应充分利用 securityContext 来约束容器权限。例如可以制定Pod安全策略（Pod Security Policies，已被Pod Security Standards取代）来禁止使用privileged:true，限制可添加的Capabilities列表。这确保集群中运行的容器不会拥有超出预期的特权。

**实践示例：** 禁止所有非必要Capabilities的Pod策略：

- 开发时，可尝试在 Pod 的 securityContext 下设置 `capabilities.drop: ["ALL"]` 并根据报错添加所需Cap。这可以测试应用实际需要哪些权限。例如某容器如果没有 CAP_CHOWN 则修改文件属主操作会失败，从而明确需要添加。Snyk 的实验显示，一个Alpine容器在无Capabilities时尝试安装软件会因为无法`chmod`某些文件而失败——暗示需要 CAP_FOWNER/CAP_DAC_OVERRIDE。通过这种方式，运维人员能找出应用运行的最小权限集，然后在部署时明确写入 YAML，杜绝多余的特权。

# Capabilities 对容器的影响和命名空间作用

正如上文多次提到的，**Namespaces（命名空间）**与Capabilities一起决定了容器能影响的范围。Linux命名空间将系统资源隔离给各容器，例如网络、进程、挂载点等。这样，即便容器拥有某个Capability，其操作通常**仅在容器自己的命名空间内有效**，对宿主或其他容器无直接影响。例如：

- **文件系统命名空间：** 容器即使有 CAP_SYS_ADMIN，可以挂载文件系统，但默认只能挂载到容器自身的挂载命名空间中，不会直接篡改宿主的挂载表。然而，如果容器通过共享卷直接访问宿主文件系统设备（如`/dev/sda`），那么有CAP_SYS_ADMIN就能把它挂载进去查看。这说明隔离机制和权限必须配合使用——只给Capability而不提供不该访问的设备，风险也相对可控。反之，若**错误地共享了敏感设备或禁用了Mount隔离**（例如使用`--pid=host --privileged`可能允许访问宿主/proc等），那么CAP_SYS_ADMIN会成为破坏隔离的利器。
- **网络命名空间：** 有了 CAP_NET_ADMIN，容器可以配置网络接口、防火墙规则等，但前提是这些网络接口属于该容器的网络命名空间。默认每个容器有自己的veth接口，对应宿主网桥，容器改动路由只影响自己。但若容器使用 `hostNetwork: true`，它实际操作的是宿主网络堆栈，此时CAP_NET_ADMIN就会影响整机网络。因此Namespaces提供了**作用域隔离**：Capability赋予的权能局限在容器自己这一小块天地，不会危及全局——只要我们不主动打破隔离。
- **PID命名空间：** 容器通常只能看到和控制自己PID NS内的进程。所以有CAP_KILL/CAP_SYS_PTRACE也只能对容器内PID有效。但如果用`hostPID: true`让容器共享了宿主进程空间，那么这些权限将可以操作宿主的任意进程，造成安全问题。为此K8s默认禁止大部分Pod使用hostPID等选项，只有特权pod才能使用，并辅以策略控制。
- **IPC、UTS 等命名空间：** 类似地，CAP_IPC_OWNER 只对容器自己的IPC资源有效，CAP_SYS_TIME 修改系统时间如果有Time NS隔离将仅改变本容器内部时钟。当前Time NS并非默认启用，所以CAP_SYS_TIME一般还是影响全局。UTS NS则隔离主机名/域名，CAP_SYS_ADMIN 可在容器内调用 `sethostname` 更改容器自己的主机名而不影响宿主。

**用户命名空间**值得单独一提：开启 User NS 后，容器内的“root”可以映射为宿主上的非特权UID，从而**大幅降低Capabilities的影响范围**。在User NS容器中，即使给予CAP_SYS_ADMIN等，因其不对应真实宿主root，很多敏感操作也无法执行（如加载模块、挂载真实设备会被内核拒绝）。Docker提供了用户命名空间隔离选项（`--userns-remap`），Kubernetes目前也支持在PodSpec启用User NS（Alpha特性），这被认为是未来容器安全的重要方向之一。

总的来说，Namespaces是容器隔离的**第一道防线**，Capabilities则是**第二道防线**：即使攻击者突破了应用本身的限制取得容器内root，也因为缺乏Capabilities无法对宿主为所欲为。反过来说，如果我们给予容器过多的Capabilities，那即使有Namespaces，也可能被其找到“杠杆”撬动宿主。例如之前提到的利用 CAP_SYS_ADMIN 挂载 cgroup 文件系统逃逸，就是结合了容器对cgroup子系统的访问权限和一个内核特性进行的攻击。这提示我们：**应同时谨慎配置Namespace隔离和Capability授予**，两者相辅相成才能保障容器安全。

# 其他安全机制：Seccomp、AppArmor/SELinux、cgroups

除了Capabilities，现代容器环境依赖多层次的安全机制共同发挥作用。它们与Capabilities互相补充，进一步减少容器逃逸和权限滥用的风险：

## Seccomp 系统调用过滤

**Seccomp** (Secure Computing Mode) 是Linux内核提供的系统调用过滤机制。Docker和Kubernetes默认启用了Seccomp的**默认策略**，拦截掉了一些高危或不常用的系统调用。从Docker文档可知，默认Seccomp配置**禁止了大约44个系统调用**（在300多个可用调用中）。这些被禁用的调用多数与改变内核状态有关，例如：`mount`（挂载命名空间之外的fs，需要CAP_SYS_ADMIN）、`kexec_load`（加载替换内核，需要CAP_SYS_BOOT）、`open_by_handle_at`（结合CAP_DAC_READ_SEARCH可能绕过权限）等等。实际上，Docker的Seccomp默认策略是**阻止清单**模式：不在允许名单上的调用一律返回 *EPERM* 错误。

Seccomp与Capabilities配合良好：一些即使容器拥有Capability的操作，也可能因为Seccomp被拦截。例如，即便给了CAP_SYS_MODULE，如果Seccomp默认策略禁止了`init_module`等系统调用，容器仍无法加载内核模块；再如CAP_SYS_ADMIN允许`clone`带某些namespace flag，但默认Seccomp会拦截带有`CLONE_NEWUSER`之外flag的`clone`调用（因为Docker本身通过其他方式创建namespace）。总之，Seccomp提供**更细粒度**的控制，在Capability之下再加一道筛选。

Kubernetes从1.19起也支持为Pod设置 seccompProfile（如 `runtime/default` 使用容器运行时的默认配置，即Docker默认策略）。通过Pod安全设置可强制要求容器运行时使用Seccomp默认或自定义策略，禁止用户将其设置为Unconfined。这样一来，即使某容器必须赋予较高Capability（如CAP_SYS_ADMIN），Seccomp仍可阻断其中最危险的一些系统调用组合，降低攻击面。

## AppArmor/SELinux 强制访问控制

**AppArmor** 和 **SELinux** 是常见的Linux强制访问控制 (MAC) 系统，作用是在内核级别进一步限制进程的行为。Docker在支持AppArmor的发行版（如Ubuntu）上，会自动为容器加载名为`docker-default`的AppArmor安全配置文件。该默认配置较为宽松以确保兼容性，但仍禁止了一些明显危险的操作，例如：禁止容器对`/proc`、`/sys`等敏感路径的写入，限制网络原始套接字，禁止 ptrace 其他进程等。如果需要，更可为特定容器制定自定义AppArmor配置，通过 `--security-opt apparmor=profile_name` 来应用。

**SELinux** 在Red Hat系发行版中用于隔离容器。Docker/Containerd会为每个容器分配一个安全上下文 (例如 `system_u:system_r:svirt_lxc_net_t:s0:c123,c456`)，所有容器内文件通常标记为 `container_file_t` 类型，对应上述的`c123,c456`类别标签。不同容器具有不同的MCS类别，从而**即使一个容器突破文件系统隔离，也无法读取另一个容器的文件**（因为标签不匹配被SELinux拒绝）。即便在特权模式下，SELinux仍对容器有限制，例如默认policy会阻止privileged容器访问宿主上的某些socket文件，除非管理员开启特定布尔值。这说明SELinux在底层为容器提供**第二重隔离**：哪怕容器拥有CAP_DAC_OVERRIDE等可以绕过文件权限，它仍无法访问宿主或其他容器文件，因为MAC策略不允许。

管理员可以利用这些MAC机制加强容器安全：

- 在Kubernetes中，通过 Pod 的 `securityContext.appArmorProfile` 指定自定义AppArmor配置；
- 使用OpenShift这样的平台则默认强制SELinux隔离容器，并提供策略以定义容器可访问的资源范围。

## cgroups 控制组隔离

**cgroups** (Control Groups) 主要用于**限制容器对系统资源的使用**。它在权限方面的作用是限制容器可以访问的设备及数量。例如Docker默认的设备 cgroup 策略只允许容器访问一小部分设备文件（如 `/dev/null`, `/dev/random` 等常见设备）以及其绑定的匿名设备（如容器自己的网络接口），禁止直接访问磁盘、物理硬件设备等。即使容器具有 CAP_MKNOD/CAP_SYS_RAWIO 之类权限，cgroup仍会阻挡其对受限设备的I/O操作。这有效防止容器通过创建设备文件直接读写宿主磁盘等情况发生。

此外，cgroups对CPU、内存的限额可防止容器滥用CAP_SYS_NICE或CAP_IPC_LOCK锁定所有内存导致宿主崩溃。比如，CAP_SYS_NICE允许提高进程优先级，但容器CPU配额只有设定的那么多，无法饿死整个系统；CAP_IPC_LOCK允许锁定内存页，但容器本身内存限制放在那里，进程顶多耗完自己的配额，不会吃光宿主RAM。因此，cgroups提供了**资源维度**的安全保障，是容器对抗DoS的一道重要防线。

# 小结

通过Capabilities、Seccomp、MAC、cgroups、Namespaces等协同，容器才能实现接近虚拟机级别的强隔离。简而言之：

- **Namespaces**划定了容器权限作用域，没有Namespace隔离，再少的Capability也可能直接影响宿主；
- **Capabilities**限定了容器在其作用域内能做的事，没有不必要的Capability，攻击者即使进入容器也缺乏进一步突破的手段；
- **Seccomp**过滤掉危险接口，哪怕有Capability也调用不了关键系统调用；
- **AppArmor/SELinux**设置禁区和额外规则，即使Capability和Syscall都放行，内核还能基于安全策略最后把关；
- **cgroups**控制资源和设备访问，避免滥用特权搞垮系统或直接操作硬件。

多层防护共同筑起容器安全堡垒。只有在特权容器(几乎关闭所有限制)的情况下，这些防线才会全面失效。因此，在设计容器部署时，应尽可能避免使用 `privileged`，而是结合上述机制按需开放最小权限。对于每一个Capability的授予，都应评估是否真有必要，以及是否有其他机制可以代替。正如业界常言：“**不可信的容器不应被赋予可信的权限**”。通过严谨的权限控制，我们可以显著降低容器逃逸和横向移动的风险，为容器化环境提供接近虚拟机的安全隔离保障。