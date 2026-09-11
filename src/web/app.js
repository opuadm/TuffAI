"use strict";

const byId=id=>document.getElementById(id);
const ADMIN_TOKEN_STORAGE="tuffai_admin_token";
let adminAuthed=false;

function getAdminToken(){return sessionStorage.getItem(ADMIN_TOKEN_STORAGE)||"";}
function adminHeaders(extra){
    const headers=Object.assign({},extra);
    const token=getAdminToken();
    if(token)headers["Authorization"]="Bearer "+token;
    return headers;
}

async function apiError(response,fallback){
    try{
        const body=await response.json();
        return new Error(body.error?.message||body.error||fallback);
    }catch(e){
        return new Error(fallback);
    }
}

function setAdminAuthed(value){
    adminAuthed=value;
    byId("navChat").hidden=!value;
    refreshChatLock();
    if(!value)byId("chatModel").replaceChildren();
}

function switchView(viewName){
    const views=document.querySelectorAll(".view");
    const buttons=document.querySelectorAll("[data-view]");

    if(viewName==="playground"&&!adminAuthed)viewName="admin";
    views.forEach(view=>{view.hidden=view.id!==viewName});
    buttons.forEach(button=>{button.classList.toggle("active",button.dataset.view===viewName)});
    history.replaceState(null,"","#"+viewName);
    if(viewName==="playground"){
        refreshChatLock();
        byId("chatInput").focus();
    }
    if(viewName==="admin")refreshAdmin();
}

function refreshChatLock(){
    const locked=!adminAuthed;
    byId("chatLock").hidden=!locked;
    byId("chatInput").disabled=locked;
    document.querySelector("#chatForm button[type=submit]").disabled=locked;
}

async function loadModels(){
    const select=byId("chatModel");
    select.replaceChildren();
    if(!adminAuthed)return;
    const response=await fetch("/v1/models",{headers:adminHeaders()});
    if(response.status===401){handleAdminExpired();throw new Error("Admin session expired. Sign in again.");}
    const body=await response.json();
    let option;

    if(!response.ok)throw new Error(body.error?.message||"Unable to load models");
    body.data.forEach(model=>{
        option=document.createElement("option");
        option.value=model.id;
        option.textContent=model.id;
        select.append(option);
    });
    updateModelOptions();
}

function handleAdminExpired(){
    sessionStorage.removeItem(ADMIN_TOKEN_STORAGE);
    setAdminAuthed(false);
    switchView("admin");
}

function updateModelOptions(){
    const select=byId("chatModel");
    const supportsAdvanced=select.value==="TuffAI-v3";

    byId("chatEffort").disabled=!supportsAdvanced;
    if(!supportsAdvanced){
        byId("chatEffort").value="";
    }
}

function addMessage(role,text){
    const item=document.createElement("div");
    const name=document.createElement("b");
    const content=document.createElement("p");

    item.className="message "+role;
    name.textContent=role==="user"?"You":"TuffAI";
    content.textContent=text;
    item.append(name,content);
    byId("messages").append(item);
    item.scrollIntoView({block:"end",behavior:"smooth"});
    return item;
}

function addStreamingMessage(){
    const item=document.createElement("div");
    const name=document.createElement("b");
    const reasoning=document.createElement("details");
    const summary=document.createElement("summary");
    const reasoningText=document.createElement("p");
    const content=document.createElement("p");

    item.className="message assistant";
    name.textContent="TuffAI";
    reasoning.className="reasoning";
    reasoning.hidden=true;
    reasoning.open=true;
    summary.textContent="Thinking";
    reasoning.append(summary,reasoningText);
    item.append(name,reasoning,content);
    byId("messages").append(item);
    item.scrollIntoView({block:"end",behavior:"smooth"});
    return{item,reasoning,reasoningText,content};
}

function processStreamBlock(block,message){
    const line=block.split("\n").find(item=>item.startsWith("data: "));
    let payload;
    let delta;

    if(!line)return false;
    if(line.slice(6)==="[DONE]")return true;
    payload=JSON.parse(line.slice(6));
    if(payload.error)throw new Error(payload.error.message||"Request failed");
    delta=payload.choices?.[0]?.delta;
    if(!delta)return false;
    if(delta.reasoning_content){
        message.reasoning.hidden=false;
        message.reasoningText.textContent+=delta.reasoning_content;
    }
    if(delta.content)message.content.textContent+=delta.content;
    message.item.scrollIntoView({block:"end"});
    return false;
}

async function readCompletionStream(response,message){
    const reader=response.body.getReader();
    const decoder=new TextDecoder();
    let buffer="";
    let result;
    let boundary;
    let block;
    let done;

    done=false;
    while(!done){
        result=await reader.read();
        buffer+=decoder.decode(result.value||new Uint8Array(),{stream:!result.done});
        boundary=buffer.indexOf("\n\n");
        while(boundary>=0){
            block=buffer.slice(0,boundary);
            buffer=buffer.slice(boundary+2);
            if(processStreamBlock(block,message))done=true;
            boundary=buffer.indexOf("\n\n");
        }
        if(result.done)break;
    }
}

function buildRequest(text){
    const system=byId("chatSystem").value.trim();
    const temperature=byId("chatTemperature").value;
    const effort=byId("chatEffort").value;
    const messages=[];
    const request={model:byId("chatModel").value,messages,stream:true};

    if(system)messages.push({role:"system",content:system});
    messages.push({role:"user",content:text});
    if(temperature)request.temperature=Number(temperature);
    if(effort)request.reasoning_effort=effort;
    if(byId("chatWebSearch").checked)request.web_search=true;
    return request;
}

async function sendChat(event){
    const input=byId("chatInput");
    const text=input.value.trim();
    const button=event.submitter;
    let message;
    let response;

    event.preventDefault();
    if(!adminAuthed){switchView("admin");return;}
    if(!text)return;
    button.disabled=true;
    input.value="";
    addMessage("user",text);
    message=addStreamingMessage();
    try{
        response=await fetch("/v1/chat/completions",{method:"POST",headers:adminHeaders({"Content-Type":"application/json"}),body:JSON.stringify(buildRequest(text))});
        if(response.status===401){handleAdminExpired();throw new Error("Admin session expired. Sign in again.");}
        if(!response.ok)throw await apiError(response,"Request failed");
        await readCompletionStream(response,message);
        if(!message.content.textContent)message.content.textContent="No response";
    }catch(error){
        message.content.textContent=error.message;
        message.item.classList.add("error");
    }finally{
        button.disabled=false;
        input.focus();
    }
}

function copyField(event){
    const id=event.currentTarget.dataset.copy;

    navigator.clipboard.writeText(byId(id).textContent);
}

function formatTime(epoch){
    if(!epoch)return"never";
    return new Date(epoch*1000).toLocaleString();
}

async function refreshAdmin(){
    const setup=byId("adminSetup");
    const login=byId("adminLogin");
    const panel=byId("adminPanel");
    const logout=byId("adminLogout");

    setup.hidden=true;
    login.hidden=true;
    panel.hidden=true;
    logout.hidden=true;
    try{
        const statusResponse=await fetch("/api/admin/status");
        const statusBody=await statusResponse.json();
        if(statusBody.setup_required){
            setAdminAuthed(false);
            setup.hidden=false;
            return;
        }
        if(!getAdminToken()){
            setAdminAuthed(false);
            login.hidden=false;
            return;
        }
        const keysResponse=await fetch("/api/admin/keys",{headers:adminHeaders()});
        if(keysResponse.status===401){
            sessionStorage.removeItem(ADMIN_TOKEN_STORAGE);
            setAdminAuthed(false);
            login.hidden=false;
            byId("adminLoginStatus").textContent="Session expired. Sign in again.";
            return;
        }
        if(!keysResponse.ok)throw await apiError(keysResponse,"Unable to load keys");
        setAdminAuthed(true);
        panel.hidden=false;
        logout.hidden=false;
        renderKeys(await keysResponse.json());
    }catch(error){
        setAdminAuthed(false);
        login.hidden=false;
        byId("adminLoginStatus").textContent=error.message;
    }
}

function renderKeys(body){
    const tbody=byId("keysBody");
    tbody.replaceChildren();
    (body.keys||[]).forEach(key=>{
        const row=document.createElement("tr");
        const name=document.createElement("td");
        const prefix=document.createElement("td");
        const created=document.createElement("td");
        const used=document.createElement("td");
        const action=document.createElement("td");
        const button=document.createElement("button");

        name.textContent=key.name||"";
        prefix.textContent=key.key||"";
        created.textContent=formatTime(key.created);
        used.textContent=formatTime(key.last_used);
        button.type="button";
        button.className="secondary small";
        button.textContent="Revoke";
        button.addEventListener("click",()=>revokeKey(key.id));
        action.append(button);
        row.append(name,prefix,created,used,action);
        tbody.append(row);
    });
    if(!body.keys||!body.keys.length){
        const row=document.createElement("tr");
        const cell=document.createElement("td");
        cell.colSpan=5;
        cell.textContent="No API keys yet. Create one above.";
        cell.className="muted";
        row.append(cell);
        tbody.append(row);
    }
}

async function adminSetup(){
    const password=byId("setupPassword").value;
    const confirm=byId("setupPasswordConfirm").value;
    const status=byId("setupStatus");

    if(password!==confirm){status.textContent="Passwords do not match.";return;}
    if(password.length<8){status.textContent="Password must be at least 8 characters.";return;}
    status.textContent="Saving…";
    try{
        const response=await fetch("/api/admin/setup",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({password})});
        const body=await response.json();
        if(!response.ok)throw new Error(body.error||"Setup failed");
        sessionStorage.setItem(ADMIN_TOKEN_STORAGE,body.token);
        byId("setupPassword").value="";
        byId("setupPasswordConfirm").value="";
        status.textContent="";
        await refreshAdmin();
        await loadModels();
    }catch(error){
        status.textContent=error.message;
    }
}

async function adminLogin(){
    const password=byId("adminPassword").value;
    const status=byId("adminLoginStatus");

    if(!password){status.textContent="Enter the admin password.";return;}
    status.textContent="Signing in…";
    try{
        const response=await fetch("/api/admin/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({password})});
        const body=await response.json();
        if(!response.ok)throw new Error(body.error||"Login failed");
        sessionStorage.setItem(ADMIN_TOKEN_STORAGE,body.token);
        byId("adminPassword").value="";
        status.textContent="";
        await refreshAdmin();
        await loadModels();
    }catch(error){
        status.textContent=error.message;
    }
}

async function adminLogout(){
    try{
        await fetch("/api/admin/logout",{method:"POST",headers:adminHeaders()});
    }catch(e){}
    sessionStorage.removeItem(ADMIN_TOKEN_STORAGE);
    setAdminAuthed(false);
    await refreshAdmin();
}

async function createKey(){
    const name=byId("newKeyName").value.trim();
    const output=byId("newKeyOutput");

    output.hidden=true;
    try{
        const response=await fetch("/api/admin/keys",{method:"POST",headers:adminHeaders({"Content-Type":"application/json"}),body:JSON.stringify({name})});
        if(response.status===401){handleAdminExpired();throw new Error("Admin session expired. Sign in again.");}
        const body=await response.json();
        if(!response.ok)throw new Error(body.error||"Unable to create key");
        output.textContent="New key ("+(body.name||"")+"): "+body.key;
        output.hidden=false;
        byId("newKeyName").value="";
        await refreshAdminKeysOnly();
    }catch(error){
        output.textContent=error.message;
        output.hidden=false;
    }
}

async function refreshAdminKeysOnly(){
    const response=await fetch("/api/admin/keys",{headers:adminHeaders()});
    if(response.status===401){handleAdminExpired();throw new Error("Admin session expired.");}
    if(!response.ok)throw await apiError(response,"Unable to load keys");
    renderKeys(await response.json());
}

async function revokeKey(id){
    const response=await fetch("/api/admin/keys?id="+encodeURIComponent(id),{method:"DELETE",headers:adminHeaders()});
    if(response.status===401){handleAdminExpired();return;}
    await refreshAdminKeysOnly();
}

async function changePassword(){
    const current=byId("curPassword").value;
    const next=byId("newAdminPassword").value;
    const status=byId("passwordStatus");
    const payload={new_password:next};

    if(next.length<8){status.textContent="Password must be at least 8 characters.";return;}
    if(current)payload.current_password=current;
    status.textContent="Saving…";
    try{
        const response=await fetch("/api/admin/password",{method:"POST",headers:adminHeaders({"Content-Type":"application/json"}),body:JSON.stringify(payload)});
        const body=await response.json();
        if(!response.ok)throw new Error(body.error||"Unable to change password");
        if(body.token)sessionStorage.setItem(ADMIN_TOKEN_STORAGE,body.token);
        byId("curPassword").value="";
        byId("newAdminPassword").value="";
        status.textContent="Password updated.";
    }catch(error){
        status.textContent=error.message;
    }
}

async function initAuth(){
    if(!getAdminToken()){setAdminAuthed(false);return;}
    try{
        const response=await fetch("/api/admin/keys",{headers:adminHeaders()});
        if(!response.ok)throw new Error("expired");
        setAdminAuthed(true);
        await loadModels();
    }catch(e){
        sessionStorage.removeItem(ADMIN_TOKEN_STORAGE);
        setAdminAuthed(false);
    }
}

byId("baseUrl").textContent=location.origin+"/v1";
byId("curlExample").textContent=byId("curlExample").textContent.replace("localhost:18298",location.host);
byId("chatModel").addEventListener("change",updateModelOptions);
byId("chatForm").addEventListener("submit",sendChat);
byId("clearChat").addEventListener("click",()=>byId("messages").replaceChildren());
byId("goAdmin").addEventListener("click",()=>switchView("admin"));
byId("setupSave").addEventListener("click",adminSetup);
byId("adminLoginBtn").addEventListener("click",adminLogin);
byId("adminLogout").addEventListener("click",adminLogout);
byId("createKey").addEventListener("click",createKey);
byId("changePassword").addEventListener("click",changePassword);
document.querySelectorAll("[data-copy]").forEach(button=>button.addEventListener("click",copyField));
document.querySelectorAll("[data-view]").forEach(button=>button.addEventListener("click",()=>switchView(button.dataset.view)));
document.querySelector(".brand").addEventListener("click",event=>{event.preventDefault();switchView("docs")});
refreshChatLock();
switchView(location.hash==="#admin"?"admin":"docs");
initAuth().then(()=>{
    if(location.hash==="#playground"&&adminAuthed)switchView("playground");
    else if(location.hash==="#playground")switchView("admin");
});
