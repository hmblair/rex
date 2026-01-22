FROM alpine:3.19

RUN apk add --no-cache openssh bash coreutils procps

# Generate host keys
RUN ssh-keygen -A

# Create test user with password auth
RUN adduser -D -s /bin/bash test && echo "test:test" | chpasswd

# Create .rex directory for job logs
RUN mkdir -p /home/test/.rex && chown test:test /home/test/.rex

# Configure sshd
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PermitEmptyPasswords no" >> /etc/ssh/sshd_config && \
    echo "UsePAM no" >> /etc/ssh/sshd_config

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e"]
